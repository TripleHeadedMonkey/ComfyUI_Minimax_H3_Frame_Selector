from .frame_selector import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    from .minimax_offset import (
        NODE_CLASS_MAPPINGS as OFFSET_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as OFFSET_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as error:
    print(f"[Interactive Frame Selector] MiniMax scoped offset node unavailable: {error}")
else:
    NODE_CLASS_MAPPINGS.update(OFFSET_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(OFFSET_NODE_DISPLAY_NAME_MAPPINGS)

try:
    from .h3_latent_tools import (
        NODE_CLASS_MAPPINGS as LATENT_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as LATENT_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as error:
    print(f"[Interactive Frame Selector] MiniMax H3 latent tools unavailable: {error}")
else:
    NODE_CLASS_MAPPINGS.update(LATENT_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(LATENT_NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
