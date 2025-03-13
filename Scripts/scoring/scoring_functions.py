import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def scoring_topn(firm, results, targets, top_n, data):
    """Retourne le nombre de build ups présents dans les ntop résultats du modèle"""
    top_index = np.argsort(results)[-top_n:][::-1]
    top_results = data.loc[top_index, 'NAME'].tolist()
    if firm in targets.keys():
        prensent = 0
        for target in targets[firm]:
            if target in top_results:
                prensent += 1
    
        print(f"Le score du modèle pour {firm} est de {prensent}/{len(targets[firm])}, pour le top {len(top_results)} résultats")
    else:
        print(f"l'entreprise {firm} n'est pas dans la liste des entreprises mères")

def scoring_pos(firm, results, targets, data):
    """"Retourne la position des builds ups dans la liste des résultats """
    if firm in targets.keys():
        present = []
        for target in targets[firm]:
            if target in results:
                print(f"Le build up {target} est classé à la position {results.index(target)} sur {len(results)} entreprises")
                present.append(results.index(target))
            else:
                print(f"Le build up {target} n'est pas présent dans les résultats pour les filtres appliqués")

        #plots a bar plot of every similarity score, with the indexes of present in a different color
        sns.set_theme(style="whitegrid")
        sns.barplot(x=data.loc[results, 'NAME'].tolist(), y = results, palette="Blues") 
        plt.xticks(rotation=90)
        plt.show()