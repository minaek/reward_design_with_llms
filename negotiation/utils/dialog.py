# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Dialogue runner class. Implements communication between two Agents.
"""

import sys
from io import TextIOWrapper

import numpy as np
import os

import utils.data as data
import utils.domain as domain
import utils.utils as utils
from utils.metric import MetricsContainer
import base_prompts
import ground_truth_rewards
from language_models import OpenAIModel
from sl_baseline import get_data_i
from transformers import pipeline, set_seed


class DialogLogger(object):
    """Logger for a dialogue."""

    CODE2ITEM = [
        ("item0", "book"),
        ("item1", "hat"),
        ("item2", "ball"),
    ]

    def __init__(self, log_file=None, append=False):
        self.logs = []
        if not os.path.exists(log_file[: log_file.index("/")]):
            print("Creating logs directory!")
            os.makedirs(log_file[: log_file.index("/")])
        if log_file:
            if type(log_file) == TextIOWrapper:  # Allow stdout
                self.logs.append(log_file)
            else:
                flags = "a" if append else "w"
                self.logs.append(open(log_file, flags))

    def _dump(self, s, forced=False):
        # def _dump(self, s, forced=True):
        for log in self.logs:
            print(s, file=log)
            log.flush()
        if forced:
            print(s, file=sys.stdout)
            sys.stdout.flush()

    def _dump_with_name(self, name, s, forced=False):
        self._dump("{0: <5} : {1}".format(name, s), forced=forced)

    def dump_ctx(self, name, ctx, forced=False):
        assert len(ctx) == 6, "we expect 3 objects"
        s = " ".join(
            [
                "%s=(count:%s value:%s)"
                % (self.CODE2ITEM[i][1], ctx[2 * i], ctx[2 * i + 1])
                for i in range(3)
            ]
        )
        self._dump_with_name(name, s, forced=forced)

    def dump_sent(self, name, sent, forced=False):
        self._dump_with_name(name, " ".join(sent), forced=forced)

    def dump_choice(self, name, choice, forced=False):
        def rep(w):
            p = w.split("=")
            if len(p) == 2:
                for k, v in self.CODE2ITEM:
                    if p[0] == k:
                        return "%s=%s" % (v, p[1])
            return w

        self._dump_with_name(name, " ".join([rep(c) for c in choice]), forced=forced)

    def dump_agreement(self, agree, forced=False):
        self._dump("Agreement!" if agree else "Disagreement?!", forced=forced)

    def dump_reward(self, name, agree, reward, forced=False):
        if agree:
            self._dump_with_name(name, "%d points" % reward, forced=forced)
        else:
            # self._dump_with_name(name, "0 (potential %d)" % reward, forced=forced)
            self._dump_with_name(name, "0 points", forced=forced)

    def dump(self, s, forced=False):
        # def dump(self, s, forced=True):
        self._dump(s, forced=forced)


class DialogSelfTrainLogger(DialogLogger):
    """This logger is used to produce new training data from selfplaying."""

    def __init__(self, verbose=False, log_file=None):
        super(DialogSelfTrainLogger, self).__init__(verbose, log_file)
        self.name2example = {}
        self.name2choice = {}

    def _dump_with_name(self, name, sent):
        for n in self.name2example:
            if n == name:
                self.name2example[n] += " YOU: "
            else:
                self.name2example[n] += " THEM: "

            self.name2example[n] += sent

    def dump_ctx(self, name, ctx):
        self.name2example[name] = " ".join(ctx)

    def dump_choice(self, name, choice):
        self.name2choice[name] = " ".join(choice)

    def dump_agreement(self, agree):
        if agree:
            for name in self.name2example:
                for other_name in self.name2example:
                    if name != other_name:
                        self.name2example[name] += " " + self.name2choice[name]
                        self.name2example[name] += " " + self.name2choice[other_name]
                        self._dump(self.name2example[name])

    def dump_reward(self, name, agree, reward):
        pass


class Dialog(object):
    """Dialogue runner."""

    def __init__(self, agents, args):
        # for now we only support dialog of 2 agents
        assert len(agents) == 2
        self.agents = agents
        self.args = args
        self.domain = domain.get_domain(args.domain)
        self.metrics = MetricsContainer()
        self._register_metrics()
        self.max_sentences = 100  # prevents infinite loops
        self.style = args.style
        self.model = args.model
        self.base_prompt, self.question = base_prompts.get_prompt(args.style)
        self.test_prompt = ""
        self.gpt3_answers = []
        self.ctxs = None

    def _register_metrics(self):
        """Registers valuable metrics."""
        self.metrics.register_average("dialog_len")
        self.metrics.register_average("sent_len")
        self.metrics.register_percentage("agree")
        self.metrics.register_average("advantage")
        self.metrics.register_pareto("pareto")
        self.metrics.register_time("time")
        self.metrics.register_average("comb_rew")
        for agent in self.agents:
            self.metrics.register_average("%s_rew" % agent.name)
            self.metrics.register_percentage("%s_sel" % agent.name)
            self.metrics.register_uniqueness("%s_unique" % agent.name)
            if agent.name != "Human":
                self.metrics.register_novelty(
                    "%s_novelty" % agent.name, self.args.novelty_model
                )
            self.metrics.register_diversity("%s_diversity" % agent.name)
            # text metrics
            ref_text = " ".join(data.read_lines(self.args.ref_text))
        self.metrics.register_ngram("full_match", text=ref_text)

    def _is_selection(self, out):
        return len(out) == 1 and out[0] == "<selection>"

    def show_metrics(self):
        return " ".join(["%s=%s" % (k, v) for k, v in self.metrics.dict().items()])

    def feed_context(self, ctxs, logger, forced=False):
        """
        Initialize agents by feeding in the contexts
        and initializing other dialogue-specific variables
        """
        # feed context
        for agent, ctx in zip(self.agents, ctxs):
            agent.feed_context(ctx)
            logger.dump_ctx(agent.name, ctx, forced=forced)
            s = " ".join(
                [
                    "%s=(count:%s value:%s)"
                    % (logger.CODE2ITEM[i][1], ctx[2 * i], ctx[2 * i + 1])
                    for i in range(3)
                ]
            )
            if agent.name == "Bob":
                self.test_prompt += f"{agent.name}   : {s}\n"
            else:
                self.test_prompt += f"{agent.name} : {s}\n"
        logger.dump("-" * 80, forced=forced)
        self.test_prompt += "-" * 80 + "\n"

    def choose_starting_order(self):
        """
        Choose who goes first by random
        :return:
        """
        first_agent_index = None
        if np.random.rand() < 0.5:
            writer, reader = self.agents
            first_agent_index = 0
        else:
            reader, writer = self.agents
            first_agent_index = 1
        return writer, reader, first_agent_index

    def write(self, writer, logger, forced=False):
        """
        Produces an utterance and saves necessary meta information
        """
        # produce an utterance
        out = writer.write()
        if not writer.human:
            # logger.dump_sent(writer.name, out, forced=forced)
            out_with_item_names = out[0].replace("item0", "book")
            out_with_item_names = out_with_item_names.replace("item1", "hat")
            out_with_item_names = out_with_item_names.replace("item2", "ball")
            if not self._is_selection(out):
                logger.dump_sent(writer.name, out, forced=forced)
                if writer.name == "Bob":
                    self.test_prompt += f"{writer.name}   : {out_with_item_names}\n"
                else:
                    self.test_prompt += f"{writer.name} : {out_with_item_names}\n"

        self.metrics.record("sent_len", len(out))
        self.metrics.record("full_match", out)
        self.metrics.record("%s_unique" % writer.name, out)
        self.metrics.record("%s_diversity" % writer.name, out)
        self.metrics.record("%s_novelty" % writer.name, out, writer)

        # append the utterance to the conversation
        self.conv.append(out)
        if len(self.conv) == 1 and ("propose" not in out[0] and "insist" not in out[0]):
            print("started conv with non-proposal : ", out)
            raise ValueError
        self.agent_order.append(writer.name)
        return out

    def read(self, reader, out):
        """
        The other agent reads the writer's utterance
        """
        # make the other agent to read it
        reader.read(out)

    def is_end(self, out, writer, reader):
        """
        Check whether the end of the conversation has been reached
        """
        # check if max_sentences has been generated
        self.num_sentences += 1
        if self.num_sentences >= self.max_sentences:
            return True
        # check if the end of the conversation was generated
        if self._is_selection(out):
            self.metrics.record("%s_sel" % writer.name, 1)
            self.metrics.record("%s_sel" % reader.name, 0)
            return True
        return False

    def generate_choices(self, ctxs, agents, logger, forced):
        """
        Generate final choices for each agent
        """
        choices = []
        # generate choices for each of the agents
        for agent in agents:
            choice = None
            if agent.name == "Alice" or agent.name == "Bob" or agent.name == "Expert":
                choice = agent.choose(self.conv, self.agent_order, ctxs)
            elif agent.name == "Human":
                choice = agent.choose()
            choices.append(choice)
        return choices

    def evaluate_choices(self, choices, ctxs, update, logger, forced, training):
        """
        Evaluate the choices, produce agreement and a reward
        :return:
        """
        # evaluate the choices, produce agreement and a reward
        agree, rewards = self.domain.score_choices(choices, ctxs)
        assert len(rewards) == 2  # Otherwise, the partner_reward will be incorrect
        logger.dump("-" * 80, forced=forced)
        # self.test_prompt += f"Alice : book={choices[0][0][-1]} hat={choices[0][1][-1]} ball={choices[0][2][-1]}\n"
        # self.test_prompt += f"Bob   : book={choices[0][3][-1]} hat={choices[0][4][-1]} ball={choices[0][5][-1]}\n"
        self.test_prompt += "-" * 80 + "\n"
        logger.dump_agreement(agree, forced=forced)
        self.test_prompt += "Agreement!\n" if agree else "Disagreement?!\n"
        for i, (agent, reward) in enumerate(zip(self.agents, rewards)):
            if agent.name == "Bob":
                self.test_prompt += f"{agent.name}   : {reward} points\n"
            else:
                self.test_prompt += f"{agent.name} : {reward} points\n"

        ## add gpt3 style reward
        if training:
            # GPT3 REWARDS
            if self.model == "gpt3":
                style_rewards = self.gpt3_reward()
                if style_rewards == -1:
                    return -1, -1

            # GPT2 REWARDS
            elif self.model == "gpt2":
                style_rewards = self.gpt2_reward()
                if style_rewards == -1:
                    return -1, -1

            # SL BASELINE REWARDS
            elif self.model == "sl_baseline":
                style_rewards = self.sl_baseline_reward()

            # GROUND TRUTH REWARDS
            elif self.model == "rl":
                if self.style == "versatile":
                    style_rewards = ground_truth_rewards.versatile(
                        self.conv, self.agent_order
                    )
                elif self.style == "pushover":
                    style_rewards = ground_truth_rewards.pushover(rewards)
                elif self.style == "competitive":
                    style_rewards = ground_truth_rewards.competitive(rewards)
                elif self.style == "stubborn":
                    style_rewards = ground_truth_rewards.stubborn(
                        self.conv, self.agent_order
                    )
                else:
                    raise ValueError
            else:
                raise ValueError

            rewards[0] = style_rewards[0]
            rewards[1] = style_rewards[1]

        pareto = 0.0
        if agree:
            self.metrics.record("advantage", rewards[0] - rewards[1])
            pareto = self.metrics.record_pareto("pareto", ctxs, rewards)
        self.update_agents(agree, rewards, update, logger, forced=forced, pareto=pareto)
        self.metrics.record("time")
        self.metrics.record("dialog_len", len(self.conv))
        self.metrics.record("agree", int(agree))
        self.metrics.record("comb_rew", np.sum(rewards) if agree else 0)
        for agent, reward in zip(self.agents, rewards):
            self.metrics.record("%s_rew" % agent.name, reward if agree else 0)
            self.metrics.record_end_of_dialogue("%s_diversity" % agent.name)
        return agree, rewards

    def sl_baseline_reward(self):
        self.test_prompt += self.question
        path = f"trained_models/sl_baseline/{self.style}.th"
        model = utils.load_model(path, cuda=False)
        model.eval()
        chunk = self.test_prompt.split("\n")[1:]
        data_i = get_data_i(chunk, extract_label=False)
        for otc in data_i["outcome"]:
            if int(otc) > 10:
                return [0, 0]
        out = model.forward(
            ctx=data_i["ctx"],
            inputs=data_i["inputs"],
            outcome=data_i["outcome"],
        )
        label = out.argmax().item()
        parsed_response = "no" if label == 0 else "yes"
        self.gpt3_answers.append((self.test_prompt, parsed_response, label))
        return [label * 10, label * 10]

    def gpt3_reward(self):
        self.test_prompt += self.question
        final_prompt = self.base_prompt + self.test_prompt
        api_key = os.environ.get("OPENAI_API_KEY")
        lm = OpenAIModel(api_key)
        response = lm.predict_token(final_prompt)
        parsed_response = response.lower().strip().split(" ")[0]
        if "no" in parsed_response:
            self.gpt3_answers.append((self.test_prompt, parsed_response, 0))
            return [0, 0]
        elif "yes" in parsed_response:
            self.gpt3_answers.append((self.test_prompt, parsed_response, 1))
            return [10, 10]
        else:
            print(f"cannot parse lm answer!: {response}")
            return -1

    def gpt2_reward(self):
        self.test_prompt += self.question
        final_prompt = self.base_prompt + self.test_prompt
        lm = pipeline("text-generation", model="gpt2")
        set_seed(10)
        full_response = lm(final_prompt, max_length=750, num_return_sequences=1)[0][
            "generated_text"
        ]
        response = full_response[len(final_prompt) :]
        parsed_response = response.lower().strip().split(" ")[0]
        if "no" in parsed_response:
            self.gpt3_answers.append((self.test_prompt, parsed_response, 0))
            return [0, 0]
        elif "yes" in parsed_response:
            self.gpt3_answers.append((self.test_prompt, parsed_response, 1))
            return [10, 10]
        else:
            print(f"cannot parse lm answer!: {response}")
            return -1

    def run(self, ctxs, logger, update, forced=False, training=True):
        """
        Run one episode of conversation
        """
        assert len(self.agents) == len(ctxs)
        self.ctxs = ctxs
        assert len(ctxs[0]) == len(ctxs[1]) == 6
        self.feed_context(ctxs, logger, forced=forced)
        writer, reader, first_agent_index = self.choose_starting_order()

        # initialize dialogue-dependent variables
        self.conv, self.agent_order = [], []
        self.metrics.reset()
        self.num_sentences = 0  # reset num_sentences
        skip = 0

        while True:
            # produce an utterance
            out = self.write(writer, logger, forced=forced)

            # make other agent read
            self.read(reader, out)

            # check whether dialogue should end
            if self.is_end(out, writer, reader):
                break

            # swap roles
            writer, reader = reader, writer

        # generate final choices for each agent
        choices = self.generate_choices(ctxs, self.agents, logger, forced=forced)

        # evaluate the choices, produce agreement and a reward
        agree, rewards = self.evaluate_choices(
            choices, ctxs, update, logger, forced, training
        )
        if agree == -1:
            skip = 1

        return skip

    def update_agents(
        self, agree, rewards, update, logger=None, forced=False, pareto=None
    ):
        """Performs update, in case if any of the agents are learnable"""
        for i, (agent, reward, update_agent) in enumerate(
            zip(self.agents, rewards, update)
        ):
            if logger is not None:
                logger.dump_reward(agent.name, agree, reward, forced=forced)
            if update_agent:
                partner_reward = rewards[-i - 1]  # 0 -> 1, 1 -> 0
                agent.update(agree, reward, partner_reward, pareto)
