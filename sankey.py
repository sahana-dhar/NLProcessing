"""
File: sankey.py
Description: A starter library that uses plotly to make sankey diagrams
Name: Jeremiah Payeur
Date - 9/26/23
"""
import plotly.graph_objects as go
import pandas as pd


def _code_mapping(df, source, target):
    """
    parameters:
        df - dataframe
        source - string source column
        target - string target column
    returns:
        dataframe - datafram
        labels - a list of all sankey source and target labels

    Note - Credit to Professor Rachlin for making this function during class time
    """
    # get distinct labels from source target columns
    labels = list(set(list(df[source]) + list(df[target])))
    # generate n intergers for n labels
    codes = list(range(len(labels)))
    # create a map from label to code
    lc_map = dict(zip(labels, codes))

    # substitute names for codes in the dataframe
    df = df.replace({source: lc_map, target: lc_map})
    return df, labels

def capitalize_string(x):
    """
    parameter:
        x: variable taken from df column row entry
    return: x as either a capitalized string or its original type
    """
    if type(x) is str:
        return x.lower().capitalize()
    else:
        return x

def dataframe_stack(df, *args):
    """
    Takes a df and an arbitrary number of its columns and stacks the dataframe
    args:
        df (pandas dataframe):
        *args (strings): column names within the dataframe
    returns:
        new_df (pandas dataframe): a new dataframe with only source target and value columns
    """
    new_df = pd.DataFrame({'Source': [], 'Target': [], 'Value': []})
    for i in range(len(args) - 1):
        temp_df = df[[args[i], args[i + 1], 'Value']]
        temp_df = temp_df.rename(columns={args[i]: 'Source', args[i + 1]: 'Target', 'Value': 'Value'})
        new_df = pd.concat([new_df, temp_df])

    return new_df

def aggregate_dataframe(df, threshold):
    """
    aggregate a dataframe that only has source target and value columns
    args:
        df (pandas dataframe): a dataframe with only source target and value columns
        threshold: only include value that are above the threshold

    returns: aggregated data and labels a list of all targets and labels
    """
    df['Value'] = df['Value'].apply(lambda x: float(x))
    agg_df = df.groupby(['Source', 'Target'])['Value'].sum().reset_index()
    agg_df = agg_df[(agg_df['Source'] != 0) & (agg_df['Value'] > threshold) & (agg_df['Target'] != 0)]
    agg_df, labels = _code_mapping(agg_df, 'Source', 'Target')
    return agg_df, labels

def make_sankey(df, *args, values=None, threshold=75, title_text=None, show=True):
    """
    parameters:
        df - dataframe
        *args - strings that represent column names to be represented in sankey diagram
        values - string, the column of the dataframe values are stored in
        threshold - int, keep only values higher than threshold
        title_text  - string, allows user to title sankey diagram

    returns:
        nothing - plots sankey diagram using plotly
    """

    if len(args) < 2:
        'Print you need at least two columns to make a sankey diagram'
        return

    if values is None:
        df['Value'] = 1
    else:
        df['Value'] = df[values]

    # create new source target value dataframe
    df = dataframe_stack(df, *args)

    # allow capitalization to not matter
    df = df.dropna()
    df['Source'] = df['Source'].apply(lambda x: capitalize_string(x))
    df['Target'] = df['Target'].apply(lambda x: capitalize_string(x))

    # aggregate data frame
    agg_df, labels = aggregate_dataframe(df, threshold)

    # Semi randomly generate colors for the links and nodes, so all links flowing out of same node are same color
    agg_df['Link_Color'] = agg_df['Source'].apply(lambda x: f'rgba({(x + 17) ** 2 % 256}, {x % 256},'
                                                            f' {(x + 15) ** 3 % 256}, .4)')
    agg_df['Node_Color'] = agg_df['Source'].apply(lambda x: f'rgba({(x + 17) ** 2 % 256}, {x % 256}, '
                                                            f'{(x + 15) ** 3 % 256}, .6)')

    # plot sankey
    link = {'source': agg_df['Source'], 'target': agg_df['Target'], 'value': agg_df['Value'],
            'color': agg_df['Link_Color']}

    node = {'label': labels, 'color': agg_df['Node_Color'], 'pad': 50, 'line': {'width': 4}}

    sk = go.Sankey(link=link, node=node)
    fig = go.Figure(sk)
    if title_text is not None:
        fig.update_layout(title_text=title_text)

    if show:
        fig.show()
        return

    else:
        return fig


def main():
    df = pd.read_csv('bio.csv')  # if in data folder can say data/bio_encoded.csv
    make_sankey(df, 'organ', 'gene', values='npub')


if __name__ == '__main__':
    main()