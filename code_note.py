Code_Note

class ShapeSpec(namedtuple("_ShapeSpec", ["channels", "height", "width", "stride"])):
"""
    Attributes:
        channels:
        height:
        width:
        stride:
"""


# Initialize model and optimizer inside
class DefaultTrainer():
"""
detectron2/engine/defaults.py
Compared to `SimpleTrainer`, it contains the following logic in addition:

    1. Create model, optimizer, scheduler, dataloader from the given config.
    2. Load a checkpoint or `cfg.MODEL.WEIGHTS`, if exists, when
       `resume_or_load` is called.
    3. Register a few common hooks.

Examples:
    .. code-block:: python

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
        trainer.train()

    Attributes:
        scheduler:
        checkpointer (DetectionCheckpointer):
        cfg (CfgNode):
"""
    model = self.build_model(cfg)
    optimizer = self.build_optimizer(cfg, model)
    data_loader = self.build_train_loader(cfg)
    
class DefaultPredicter
"""
detectron2/engine/defaults.py

Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

Examples:

    .. code-block:: python

        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
"""