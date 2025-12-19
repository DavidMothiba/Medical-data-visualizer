import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 import the data
df = pd.read_csv("medical_examinaion.csv")

# 2 add an overweight column
df['overweight'] = (df['overweight']/((df['height']/100)**2)>25).astype(int)

# 3 Normalize cholesterol and glucose (0 = good, 1 = bad)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


# 4 draw categorical plot
def draw_cat_plot():
    
    # 5. creae Dataframe for categorcal plot
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group and reformat the data 
    df_cat = (
        df_cat
        .groupby(['cardio', 'variable', 'value'])
        .size()
        .reset_index(name='total')
    )
    
    # 7.
    # 8. Creae the catplot
    fig = sns.catplot(
        data=df_cat,
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        kind='bar'
    ).fig

    # 9. Sav the igure
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11. Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12. Calculate correlation matrix
    corr = df_heat.corr()

    # 13. Generate mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15. Draw heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        cbar_kws={'shrink': 0.5}
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
