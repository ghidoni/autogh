import pandas as pd
import numpy as np

lista_normal = np.random.standard_normal(1000)
lista_normal = (lista_normal - np.min(lista_normal)) / (
    np.max(lista_normal) - np.min(lista_normal)
)
df = pd.DataFrame(
    {
        "id": np.arange(0, 1000),
        "score": lista_normal,
        "label": np.random.randint(0, 2, 1000),
    }
)

df.to_csv("pygh_data.csv", index=False)