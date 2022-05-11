import plotly.express as px
import pandas as pd
from result import synthesized_as_support_result as result

l = []
for model in result.keys():
    if model == "Radar":
        continue
    accs = result[model]
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

fig.write_image("./output/support_from_synthesized.png", format="png", width=700, height=500)
# fig.show()
