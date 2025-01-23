from phi.tools import Toolkit
from phi.utils.log import logger

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class RandomForest(Toolkit):
    def __init__(self):
        super().__init__(name="random_forest")
        self.register(self.run_random_forest)

    def run_random_forest(self, args) -> dict:
        """
        Runs a random forest algorithm using the DataFrame provided in args. The DataFrame should have a target variable 
        named 'target'. Output will be a classification matrix in a dictionary.

        Args:
            args: Input which is expected to be a Pandas DataFrame or a compatible format.
        Returns:
            dict: The output is a dictionary containing the classification matrix.
        """
        logger.info(f"Running Random Forest with args: {args}")

        # Validate and extract DataFrame from args
        if isinstance(args, pd.DataFrame):
            df = args
        elif isinstance(args, dict):
            try:
                df = pd.DataFrame(args)
            except Exception as e:
                logger.error("Failed to convert dict to DataFrame")
                raise ValueError("Provided args could not be converted to a DataFrame.") from e
        else:
            logger.error("Invalid input type. Expected DataFrame or dict.")
            raise ValueError("args must be a Pandas DataFrame or a dictionary.")

        # Check if 'target' column exists
        if 'target' not in df.columns:
            logger.error("'target' column is missing in the DataFrame.")
            raise KeyError("The DataFrame must contain a 'target' column.")

        # Split data into features and target
        X = df.drop(columns=['target'])  # Features (all columns except 'target')
        y = df['target']                # Target variable

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Initialize and train the RandomForestClassifier
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        rf_model.fit(X_train, y_train)

        # Predict and generate classification report
        y_pred = rf_model.predict(X_test)

        return classification_report(y_test, y_pred, output_dict=True)
