"""Utilities for qablet models"""

from qablet._qablet import mc_price


class MCPricer:
    """MC Pricer that uses a Py Model."""

    def __init__(self, state_class):
        self.state_class = state_class

    def price(self, timetable, dataset):
        """Calculate price of contract.

        Parameters:
            timetable (dict): timetable for the contract.
            dataset (dict): dataset for the model.

        Returns:
            price (float): price of contract
            stats (dict): stats such as standard error

        """

        model_state = self.state_class(dataset)
        model_state.reset()
        price = mc_price(
            timetable["events"],
            model_state,
            dataset,
            timetable.get("expressions", {}),
        )

        return price, model_state.stats
