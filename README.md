# Hands on Machine Learning

A hands-on introduction to core machine learning techniques. In the guided notebook you will scale features, tune hyperparameters, and compare models with cross-validation on Titanic data. In the exercise you join a heart-related teaching dataset from PostgreSQL, check its target coding and data quality, and build a classifier for recorded disease. Finally, reload the saved Titanic pipeline, validate new inputs, and deliver predictions with an input-quality report.

## Learning Objectives

By the end of this repository, you should be able to:

- Follow the machine learning workflow from defining a goal to evaluating a model.
- Scale features with standardization and min-max scaling, and distinguish these from row normalization.
- Validate models reliably using K-fold cross-validation.
- Tune hyperparameters with grid search and randomized search.
- Fetch and join data from a SQL database to assemble a training set.
- Check a dataset's source, feature definitions, missing values, and target coding before modelling.
- Apply the full workflow end to end to build a classifier on a new problem.
- Save and reload a fitted pipeline, validate new inputs, and interpret batch predictions.

## Learning Path

> [!TIP]
> Start with the [**Machine Learning Workflow**](machine_learning_workflow.md) guide. It walks through every step of an ML project (define the goal, get the data, split, explore, model, evaluate) and gives you the map for what the notebooks practice.

The notebooks build on each other in order:

| File / Folder | Description |
|---|---|
| [**The Machine Learning Workflow**](machine_learning_workflow.md) | Step-by-step overview of an end-to-end ML project. Read this first. |
| [**1 - Scaling & Hyperparameter Tuning**](1_scaling_hyperparameter_tuning.ipynb) | Establish a majority-class baseline, compare unscaled and scaled models with the same CV folds, tune KNN, then evaluate once on test data. |
| [**2 - Exercise: Machine Learning**](2_exercise_machine_learning.ipynb) | Join five PostgreSQL tables, correct the inverted target, handle quality issues, and practice the complete classification workflow in seven guided stages. |
| [**3 - From Training to Predictions**](3_from_training_to_predictions.ipynb) | Reload the frozen Titanic pipeline, validate invented requests, predict by batch, investigate sensitivity, and prepare a handover. Includes six exercises. |

For notebook 3, run notebook 1 through **Export for Notebook 3** first.
The saved pipeline loads in a fresh kernel. Keep its configuration fixed;
the new invented requests have no true labels and cannot establish accuracy.

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | Titanic files and an invented prediction batch for notebook 3. The older preprocessed Titanic CSV is retained for reference only; notebook 2 reads PostgreSQL. |
| [**Assets**](assets/) | Images used in the notebooks and the workflow guide. |
| [**Solutions**](solutions/) | Reference solutions. |
| [**.env.example**](.env.example) | Database credential template for notebook 2; copy to a local `.env`. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

The notebooks create a local, Git-ignored `artifacts/` directory for the saved
Titanic pipeline, metadata, prediction reports, and your model handover.
After completing the exercises, use the worked solutions for
[notebook 2](solutions/2_exercise_machine_learning.ipynb) and
[notebook 3](solutions/3_from_training_to_predictions.ipynb) to compare your choices.

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it including the `< >` brackets with your own value. For example, `cd <repo-name>` becomes `cd ds-hands-on-ml`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in (`.venv/`).

```bash
cd <repo-name>
uv sync
```

---

### 5. Set Up the Database Connection

Notebook 2 and its solution use the shared PostgreSQL database.
Ask your coach for connection details and read access to the `heart` schema.
Notebooks 1 and 3 use the included Titanic files and need no credentials.

Copy the template:

```bash
cp .env.example .env
```

Fill in `DATABASE`, `USER_DB`, `PASSWORD`, `HOST`, and `PORT`.
The notebook builds the URL automatically; enter the password as-is,
without URL encoding. An existing complete `DB_STRING` is also supported.

The database is a teaching adaptation, not an exact copy of the original UCI
file. Its `heart_attack` field is inverted: **1 means no disease recorded,
0 means disease present**. Notebook 2 explains how to preserve that field and
derive `disease_present = 1 - heart_attack`, with class 1 meaning disease.
This is not future-heart-attack prediction.

> [!CAUTION]
> Never commit `.env` or database exports. Both are Git-ignored. The supplied notebook queries use read-only transactions.

---

### 6. Open the Notebooks

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open a notebook and select the Python environment created by `uv sync` as the kernel.

## References & Further Reading

- [**UCI Heart Disease**](https://archive.ics.uci.edu/dataset/45/heart+disease): Original Cleveland data, attribute definitions, and dataset license.
- [**The Machine Learning Workflow**](machine_learning_workflow.md): The step-by-step guide in this repo, your starting point.
- [**Feature Engineering**](https://github.com/neuefische/ds-feature-engineering): Companion bootcamp repo to go deeper on feature engineering (imputation, encoding, binning, feature creation).
- [**Scikit-learn: Preprocessing data**](https://scikit-learn.org/stable/modules/preprocessing.html): Standardization, normalization, and other scaling methods.
- [**Scikit-learn: Cross-validation**](https://scikit-learn.org/stable/modules/cross_validation.html): Evaluating estimator performance and avoiding overfitting.
- [**Scikit-learn: Tuning hyperparameters**](https://scikit-learn.org/stable/modules/grid_search.html): Grid search, randomized search, and successive halving.
- [**Scikit-learn: Model persistence**](https://scikit-learn.org/stable/model_persistence.html): Saving and reloading fitted models and their environments.
