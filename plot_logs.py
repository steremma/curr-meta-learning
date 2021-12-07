import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

vanilla = pd.read_csv("vanilla.log.csv")
variant = pd.read_csv("curriculum.log.csv")

fig = plt.figure(figsize=(10, 7))
sns.lineplot(data=variant, x="iteration", y="valid_error", label="Curriculum")
sns.lineplot(data=vanilla, x="iteration", y="valid_error", label="Vanilla")
plt.axvline(x=30000, linestyle="dashed", alpha=0.5)
plt.axvline(x=7500, linestyle="dotted")
plt.axvline(x=15000, linestyle="dotted")
plt.axvline(x=22500, linestyle="dotted")

plt.ylabel('Training Accuracy')
plt.xlabel('Iteration')
plt.show()
plt.close()


fig = plt.figure(figsize=(10, 7))
sns.lineplot(data=variant, x="iteration", y="train_accuracy", label="Curriculum")
sns.lineplot(data=vanilla, x="iteration", y="train_accuracy", label="Baseline")
plt.axvline(x=30000, linestyle="dashed", alpha=0.5)
plt.axvline(x=7500, linestyle="dotted")
plt.axvline(x=15000, linestyle="dotted")
plt.axvline(x=22500, linestyle="dotted")
plt.ylabel('Training Accuracy')
plt.xlabel('Iteration')
plt.show()
plt.close()

fig = plt.figure(figsize=(10, 7))
sns.lineplot(data=variant, x="iteration", y="valid_accuracy", label="Curriculum")
sns.lineplot(data=vanilla, x="iteration", y="valid_accuracy", label="Baseline")
plt.ylabel('Validation Accuracy')
plt.xlabel('Iteration')
plt.show()
plt.close()
