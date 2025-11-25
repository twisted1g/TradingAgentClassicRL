from .sarsa_lambda_agent import SarsaLambdaAgent


class SarsaAgent(SarsaLambdaAgent):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            lambda_param=0,
            replace_traces=True,
            name="SARSA",
            **kwargs,
        )
