import unittest

from text.text import (remove_url, remove_emoticons, tokenize_text,
                       untokenize_text, get_text_cloud, get_freq_dist_list)


def crop_statements_until_t(df, t):
    
    df = df[df.time<t]
    idspoliticos = df.id_politico.unique();
    tau = []
    
    for i in idspoliticos:
        taui = df.isFavorable[df.id_politico == i]
        tau.append([taui.tolist(),i])
    
    return tau