from diffusion_policy.env.kitchen.base import KitchenBase


class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner", "light switch"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenKettleMicrowaveLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "microwave", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenAllV0(KitchenBase):
    TASK_ELEMENTS = KitchenBase.ALL_TASKS
