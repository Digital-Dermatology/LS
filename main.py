from fastai.vision.all import *

data_path="/DATA"
model_path="/MODEL"

torch.cuda.set_device(0)

dls = ImageDataLoaders.from_folder(data_path, train="train", valid="test", seed=42, bs=128, item_tfms=Resize(480),
                                    batch_tfms=aug_transforms(size=224))

learn = cnn_learner(dls, resnet34, metrics=[accuracy, Precision(), Recall(), RocAucBinary()], path=model_path)
learn.to_fp16()

learn.fine_tune(10, 1e-2, freeze_epochs=5)

learn.save('LS_detector_binary_both')