def trouve_data_case(df, pos, latitude_min, longitude_min, ecart_x, ecart_y):
    """DataFrame * (int,int) * float * float * flot * float -> DataFrame
        Retourne un DataFrame contenant toutes les lignes se situant dans la case pos.
    """
    x, y = affectation_2(df, latitude_min, longitude_min, ecart_x, ecart_y)
    i, j = pos
    return df[(x==i) & (y==j)]


