def versatile(conv, agent_order):
    """Alice is versatile if she does not suggest the same proposal more than once and suggests no other proposal."""
    assert len(conv) == len(agent_order)
    alice_utterances = []
    for i in range(len(conv)):
        utterance = conv[i][0]
        if agent_order[i] == "Alice" and "selection" not in utterance:
            alice_utterances.append(utterance)
    assert len(alice_utterances) > 0

    # remove copies of the same proposal, count propose and insist the same
    count = {}
    dup_alert = False
    if len(alice_utterances) > 1:
        x = 4
    for utt in alice_utterances:
        utt_list = utt.split(" ")
        if len(utt_list) > 1:
            assert "propose" in utt_list[0] or "insist" in utt_list[0]
            utt = " ".join(utt_list[1:])
            if utt in count:  # there is a dup!
                dup_alert = True
            else:
                count[utt] = 1

    if len(alice_utterances) > 1 and dup_alert:
        return [0, 0]
    elif len(alice_utterances) == 1:
        return [0, 0]
    else:
        return [10, 10]


def pushover(original_rewards):
    if original_rewards[0] < original_rewards[1]:
        return [10, 10]
    else:
        return [0, 0]


def competitive(original_rewards):
    if original_rewards[0] > original_rewards[1]:
        return [10, 10]
    else:
        return [0, 0]


def stubborn(conv, agent_order):
    assert len(conv) == len(agent_order)
    alice_utterances = []
    for i in range(len(conv)):
        utterance = conv[i][0]
        if agent_order[i] == "Alice" and "selection" not in utterance:
            alice_utterances.append(utterance)
    assert len(alice_utterances) > 0

    # remove copies of the same proposal, count propose and insist the same
    count = {}
    dup_alert = False
    for utt in alice_utterances:
        utt_list = utt.split(" ")
        if len(utt_list) > 1:
            assert "propose" in utt_list[0] or "insist" in utt_list[0]
            utt = " ".join(utt_list[1:])
            if utt in count:  # there is a dup!
                dup_alert = True
            else:
                count[utt] = 1
    if len(alice_utterances) > 1 and dup_alert:
        return [10, 10]
    else:
        return [0, 0]
