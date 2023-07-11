

def crop_statements_until_t(df, t):
    
    df = df[df.time<t]
    idspoliticos = df.Id_politico.unique()
    tau = []
    
    for i in idspoliticos:
        taui = df.isFavorable[df.Id_politico == i]
        tau.append([taui.tolist(),i])    #inclui iD politico
    
    return tau