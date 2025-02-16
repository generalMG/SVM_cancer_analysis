from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

from sklearn.svm import SVC

cancer_data = load_breast_cancer()
X = cancer_data.data[:, :2]
Y = cancer_data.target

SVM = SVC(kernel="rbf", gamma=0.5, C=1.0)

SVM.fit(X, Y)

DecisionBoundaryDisplay.from_estimator(
    SVM,
    X,
    response_method="predict",
    cmap=plt.cm.Spectral,
    alpha=0.8,
    xlabel=cancer_data.feature_names[0],
    ylabel=cancer_data.feature_names[1],
)

plt.scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolors='k')
plt.show()