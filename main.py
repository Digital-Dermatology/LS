from fastai.vision.all import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from fastai.vision.all import load_learner

data_path="/DATA"
model_path="/MODEL"

torch.cuda.set_device(0)


###########################################################################################################################
# Resutls of the grid search
###########################################################################################################################
lr=3e-4
epoch=10
freeze_epoch=5
batch_size=128
###########################################################################################################################

dls = ImageDataLoaders.from_folder(data_path, train="train", valid="test", seed=42, bs=batch_size, item_tfms=Resize(480),
                                    batch_tfms=aug_transforms(size=224))

learn = cnn_learner(dls, resnet34, metrics=[accuracy, Precision(), Recall(), RocAucBinary()], path=model_path)
learn.to_fp16()

learn.fine_tune(epoch, lr, freeze_epochs=freeze_epoch)

learn.save('LS_detector_binary_both')

print(interp.print_classification_report())

# Get predictions and targets for the validation set
preds, targets = learn.get_preds()

# Binarize the targets for multiclass ROC curve
# Get the number of classes and class labels from the dataloaders
n_classes = learn.dls.c
class_labels = learn.dls.vocab
targets_binarized = label_binarize(targets, classes=list(range(n_classes)))

# Plot ROC curve for each class
plt.figure(figsize=(12, 8))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(targets_binarized[:, i], preds[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.2f})')

# Plot baseline ROC curve
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')

# Formatting the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Multiclass')
plt.legend(loc="lower right")
plt.show()

# Save the plot as a PDF
plt.savefig('roc_curve_multiclass.pdf')
