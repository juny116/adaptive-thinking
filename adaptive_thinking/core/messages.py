"""Conversation formatting."""


def messages_to_string(messages):
    text = ""
    for message in messages:
        text += message["content"]
    return text
