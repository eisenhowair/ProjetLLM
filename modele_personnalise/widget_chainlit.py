from typing import Optional

import chainlit as cl


@cl.set_chat_profiles
async def chat_profile(current_user: cl.User):
    print("dans chat profile")
    if current_user.metadata["role"] != "ADMIN":
        return None

    return [
        cl.ChatProfile(
            name="ff",
            markdown_description="profil de ff",
        )
    ]


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    print("dans auth_callback")
    if (username, password) == ("admin", "admin"):
        return cl.User(identifier="admin", metadata={"role": "ADMIN"})
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    print("dans on chat start")

