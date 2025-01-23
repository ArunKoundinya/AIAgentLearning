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

    def run_random_forest(self, args: dict) -> str:
        """
        Runs a random forest algorithm using the input data or a link to a CSV file.
        Outputs a classification matrix in a String format.

        Args:
            args (dict): A dictionary containing:
                - 'data': A Pandas DataFrame or a string URL to a CSV file.
        Returns:
            str: A stringified classification matrix report.
        """
        try:
            # Validate and load data
            if isinstance(args, dict) and 'data' in args:
                if isinstance(args['data'], pd.DataFrame):
                    df = args['data']
                elif isinstance(args['data'], str):  # Assume it's a URL or file path
                    df = pd.read_csv(args['data'])
                else:
                    raise ValueError("Invalid data input. Must be a DataFrame or a CSV file URL.")
            else:
                raise ValueError("Missing 'data' key in arguments.")

            # Split data into features and target
            if 'target' not in df.columns:
                raise ValueError("'target' column missing in the dataset.")

            X = df.drop(columns=['target'])  # Features
            y = df['target']  # Target variable

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
            report = classification_report(y_test, y_pred, output_dict=True)

            return str(report)

        except Exception as e:
            logger.error(f"Error in run_random_forest: {str(e)}")
            return str({"error": str(e)})
