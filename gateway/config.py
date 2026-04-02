from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    xai_api_key: str = ""
    anthropic_api_key: str = ""
    hf_token: str = ""
    router_url: str = "http://router:8100"
    watchdog_url: str = "http://watchdog:8200"
    models_dir: str = "/models"
    shared_dir: str = "/shared"
    eve_voice: str = "eve"
    eve_reference_image: str = "eve-NATURAL.png"
    greeting_text: str = (
        "Hello! I'm Eve, your digital companion. "
        "I'm so glad you're here. What would you like to talk about?"
    )

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
