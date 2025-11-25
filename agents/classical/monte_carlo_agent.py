from .sarsa_lambda_agent import SarsaLambdaAgent


class MonteCarloAgent(SarsaLambdaAgent):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            lambda_param=1,
            replace_traces=True,
            name="MonteCarlo",
            **kwargs,
        )
