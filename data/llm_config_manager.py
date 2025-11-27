import json
import os
from typing import Dict, List, Optional, Any

from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel, Field, validator
from langchain_deepseek import ChatDeepSeek


from utils.exceptions import ConfigurationError
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIConfig(BaseModel):
    """OpenAI configuration model with validation"""
    api_key: str = Field(..., description="OpenAI API key")
    base_url: Optional[str] = Field("https://api.openai.com/v1", description="API base URL")
    model: str = Field("gpt-4", description="Model name")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperature for generation")
    max_tokens: int = Field(2000, ge=1, le=32000, description="Maximum tokens to generate")
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")

    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("API key cannot be empty")
        # Check for placeholder values
        if "your_" in v.lower() or "placeholder" in v.lower():
            raise ValueError("Please replace placeholder API key with actual value")
        return v.strip()

    @validator('base_url')
    def validate_base_url(cls, v):
        if v and not v.startswith(('http://', 'https://')):
            raise ValueError("Base URL must start with http:// or https://")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration model with validation"""
    api_key: str = Field(..., description="Embedding API key")
    base_url: Optional[str] = Field("https://api.openai.com/v1", description="API base URL")
    model: str = Field("text-embedding-3-small", description="Embedding model name")
    dimensions: Optional[int] = Field(None, ge=1, le=8192, description="Embedding dimensions")
    timeout: int = Field(30, ge=1, description="Request timeout in seconds")

    @validator('api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("API key cannot be empty")
        if "your_" in v.lower() or "placeholder" in v.lower():
            raise ValueError("Please replace placeholder API key with actual value")
        return v.strip()


class LLMConfig(BaseModel):
    """LLM configuration model"""
    provider: str = Field(..., description="LLM provider (openai, azure, etc.)")
    config: Dict[str, Any] = Field(..., description="Provider-specific configuration")

    @validator('provider')
    def validate_provider(cls, v):
        allowed_providers = ['openai', 'azure', 'anthropic', 'cohere']
        if v.lower() not in allowed_providers:
            raise ValueError(f"Provider must be one of: {', '.join(allowed_providers)}")
        return v.lower()


class LlmConfigManager:
    """Enhanced configuration manager for LLM and embedding services with environment variable support"""

    def __init__(self, config_file: str = "config/llm_config.json"):
        self.config_file = config_file
        self.llm_configs: Dict[str, LLMConfig] = {}
        self.embedding_configs: Dict[str, EmbeddingConfig] = {}
        self._load_from_environment()
        self.load_config()

    def _load_from_environment(self):
        """Load configuration from environment variables with fallback to file"""
        # Check for environment variables and override file configs
        env_api_key = os.getenv('OPENAI_API_KEY')
        env_base_url = os.getenv('OPENAI_BASE_URL')
        env_model = os.getenv('OPENAI_MODEL')

        if env_api_key:
            # Create or update default LLM config from environment
            default_config = {
                "api_key": env_api_key,
                "base_url": env_base_url or "https://api.openai.com/v1",
                "model": env_model or "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "timeout": 30
            }

            # Update or create default LLM config
            if "default" in self.llm_configs:
                self.llm_configs["default"].config.update(default_config)
            else:
                self.llm_configs["default"] = LLMConfig(
                    provider="openai",
                    config=default_config
                )

    def load_config(self):
        """Load configuration from file with enhanced error handling"""
        try:
            logger.info(f"Loading configuration from {self.config_file}")
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # Load LLM configurations with validation
            if "llm_configs" in config_data:
                for config_id, config in config_data["llm_configs"].items():
                    try:
                        # Validate provider-specific config
                        if config["provider"].lower() == "openai":
                            openai_config = OpenAIConfig(**config["config"])
                            config["config"] = openai_config.dict()

                        self.llm_configs[config_id] = LLMConfig(**config)
                        logger.info(f"Loaded LLM config: {config_id}")
                    except Exception as e:
                        logger.error(f"Invalid LLM configuration for {config_id}: {e}")
                        raise ConfigurationError(f"Invalid LLM config {config_id}: {e}")

            # Load embedding configurations with validation
            if "embedding_configs" in config_data:
                for config_id, config in config_data["embedding_configs"].items():
                    try:
                        embedding_config = EmbeddingConfig(**config)
                        self.embedding_configs[config_id] = embedding_config
                        logger.info(f"Loaded embedding config: {config_id}")
                    except Exception as e:
                        logger.error(f"Invalid embedding configuration for {config_id}: {e}")
                        raise ConfigurationError(f"Invalid embedding config {config_id}: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")


    def get_llm_config(self, config_id: str) -> Optional[LLMConfig]:
        """Get LLM configuration by ID with fallback"""
        config = self.llm_configs.get(config_id)
        if not config:
            logger.warning(f"LLM configuration not found: {config_id}")
            # Try to fallback to default if available
            if config_id != "default" and "default" in self.llm_configs:
                logger.info(f"Falling back to default LLM configuration")
                return self.llm_configs["default"]
        return config

    def get_embedding_config(self, config_id: str) -> Optional[EmbeddingConfig]:
        """Get embedding configuration by ID with fallback"""
        config = self.embedding_configs.get(config_id)
        if not config:
            logger.warning(f"Embedding configuration not found: {config_id}")
            # Try to fallback to default if available
            if config_id != "default" and "default" in self.embedding_configs:
                logger.info(f"Falling back to default embedding configuration")
                return self.embedding_configs["default"]
        return config


    def get_llm(self, config_id: str) -> ChatOpenAI:
        """Get configured LLM instance with error handling"""
        llm_config = self.get_llm_config(config_id)
        if not llm_config:
            llm_config = self.get_llm_config("default")


        conf = llm_config.config
        try:
            model = ChatOpenAI(
                model=conf["model"],
                temperature=conf["temperature"],
                max_tokens=conf["max_tokens"],
                api_key=conf["api_key"],
                base_url=conf["base_url"]
            )
            logger.info(f"Initialized LLM: {conf['model']} from {config_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to initialize LLM {config_id}: {e}")
            raise ConfigurationError(f"LLM initialization failed: {e}")

    def get_embedding(self, config_id: str, query: str) -> List[float]:
        """Get embedding for query with error handling"""
        embedding_config = self.get_embedding_config(config_id)
        if not embedding_config:
            raise ConfigurationError(f"Embedding configuration not found: {config_id}")

        try:
            model = OpenAI(
                api_key=embedding_config.api_key,
                base_url=embedding_config.base_url
            )

            response = model.embeddings.create(
                input=[query],
                model=embedding_config.model
            )

            logger.debug(f"Generated embedding for query: {query[:50]}...")
            return response.data[0].embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise ConfigurationError(f"Embedding generation failed: {e}")
