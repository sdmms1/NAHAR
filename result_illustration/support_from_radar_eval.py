import plotly.express as px
import pandas as pd
from result import system_eval_result as result

for scenario in ["scenario %d" % x for x in range(1, 5)]:
    for models in [
        # ["FSHAR", "OneFi", "Radar Model"],
        ["Few Radar", "Synthesized Model", "Fine Tune",  "Radar Model"],
        # ["FSHAR", "OneFi", "Synthesized Model", "Fine Tune", "Radar Model"]
    ]:
        l = []
        for model in models:
            accs = result[model][scenario]
            for i, acc in enumerate(accs):
                l.append([model, "Env %d" % (6 - len(accs) + i), acc])
        df = pd.DataFrame(l, columns=["Model", "Environment", "Accuracy"])
        print(df)

        fig = px.bar(
            df,
            x="Environment",
            y="Accuracy",
            color="Model",
            barmode="group",
            range_y=[0, 100]
        )

        fig.write_image("./output/eval_task2_env%s.png" % scenario[-1], format="png")
        # fig.show()
