import logging
from fastapi import FastAPI

from vocode.streaming.models.agent import ChatGPTAgentConfig
from vocode.streaming.synthesizer.eleven_labs_synthesizer import ElevenLabsSynthesizer
from vocode.streaming.synthesizer.google_synthesizer import (
    GoogleSynthesizer,
    GoogleSynthesizerConfig,
)
from vocode.streaming.synthesizer.stream_elements_synthesizer import (
    StreamElementsSynthesizer,
)
from vocode.streaming.transcriber.google_transcriber import (
    GoogleTranscriber,
    GoogleTranscriberConfig,
)
from vocode.streaming.models.synthesizer import (
    ElevenLabsSynthesizerConfig,
    StreamElementsSynthesizerConfig,
)

from vocode.streaming.agent.chat_gpt_agent import ChatGPTAgent
from vocode.streaming.client_backend.conversation import ConversationRouter
from vocode.streaming.models.message import BaseMessage

import os
import uvicorn

app = FastAPI(docs_url=None)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

GOOGLE_TRANSCRIPER_THUNK = lambda input_audio_config: GoogleTranscriber(
    GoogleTranscriberConfig.from_input_audio_config(input_audio_config)
)

STREAM_ELEMENTS_SYNTHESIZER_THUNK = (
    lambda output_audio_config: StreamElementsSynthesizer(
        StreamElementsSynthesizerConfig.from_output_audio_config(output_audio_config)
    )
)
# much more realistic, but slower responses and requires a paid API key
ELEVEN_LABS_SYNTHESIZER_THUNK = lambda output_audio_config: ElevenLabsSynthesizer(
    ElevenLabsSynthesizerConfig.from_output_audio_config(
        output_audio_config,
        api_key=os.getenv("ELEVEN_LABS_API_KEY"),
    )
)

GOOGLE_SYNTHESIZER_THUNK = lambda output_audio_config: GoogleSynthesizer(
    GoogleSynthesizerConfig.from_output_audio_config(output_audio_config)
)

conversation_router = ConversationRouter(
    agent=ChatGPTAgent(
        ChatGPTAgentConfig(
            initial_message=BaseMessage(text="Hello!"),
            prompt_preamble="Have a pleasant conversation about life",
            model_name="chatgpt-4o-latest",
        )
    ),
    transcriber_thunk=GOOGLE_TRANSCRIPER_THUNK,
    synthesizer_thunk=STREAM_ELEMENTS_SYNTHESIZER_THUNK,
    logger=logger,
)

app.include_router(conversation_router.get_router())

uvicorn.run(app, host="0.0.0.0", port=3000)
