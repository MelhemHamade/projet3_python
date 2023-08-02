#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # PROJET-3

# ## Stratégies

# ### Exploration des Données 
# 
# ### Stratégie de Nettoyage et de Filtration des Données 
# 
# ### Stratégie d'Imputation des Données
# 
# ### Exploration pour Evaluer la Qualité d'Imputation
# 
# #### Conclusions en Chiffres
# 
# ### Principes du RGPDet prendre des décisions éclairées.
# 
# le projet respecte les principes du RGPD :
# 1. Principe de licéité, loyauté et transparence :
# Les données sont collectées de leurs sources légitimes et avec leur consentement et compréhention, ainsi que la finalité de la collecte et des traitements.
# 2. Principe de limitation des finalités :
# La finalité du projet est limitée, claire et conforme aux attentes des utilisateurs.
# 3. Principe de minimisation des données :
# La collecte et les traitements des données sont optimisés.
# 4. Principe d'exactitude des données :
# Les données collectées et utilisées dans le projet sont reputées exactes, à jour et pertinentes, après nettoyage. 
# Les données collectées et utilisées dans le projet sont reputées exactes, à jour et pertinentes, après nettoyage. 
# 
# 5. Principe de limitation de la conservation :
# 

# ## Fonctions

# ### Fonction de Nettoyage & Filtrage

# In[1]:


def replace_outliers_by_NaN(dataframe, targeted_columns, n_iters):
    modified_dataframe = dataframe.copy()  # Copie du DataFrame initial

    for _ in range(n_iters):
        # Parcourir toutes les colonnes du DataFrame
        for column_name in targeted_columns:
            # Assurez-vous que la colonne est de type numérique
            if np.issubdtype(modified_dataframe[column_name].dtype, np.number):
                # Calculer l'IQR de la colonne
                Q1 = modified_dataframe[column_name].quantile(0.25)
                Q3 = modified_dataframe[column_name].quantile(0.75)
                IQR = Q3 - Q1

                # Définir la limite pour une valeur aberrante
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR
                
                # Remplacer les valeurs aberrantes par NaN dans la copie modifiée
                modified_dataframe.loc[(modified_dataframe[column_name] < lower_limit) | (modified_dataframe[column_name] > upper_limit), column_name] = np.nan

    return modified_dataframe


# In[2]:


def clean_dataframe_by_domain(df, exclude_columns):

    """
    Cette fonction supprime dans df les valeurs entre 0 et 100 dans les colonnes de la liste exclude_columns
    """
    for column in df.columns[~df.columns.isin(exclude_columns)]:
        df[column] = df[column].apply(lambda x: x if 0 <= x <= 100 else np.nan)
    return df
    
import numpy as np


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression

def clean_variables_by_participation_constraints(df, involved_variables, includer_column='includer_100g'):
    """
    Applique la fonction clean_variable_by_inclusion_constraint à plusieurs variables.

    :param df: Le DataFrame à nettoyer.
    :param participating_variables: La liste des variables à nettoyer.
    :param includer_column: Le nom de la colonne d'énergie (par défaut : 'includer_100g').
    :return: Le DataFrame nettoyé.
    """

    def get_includer_coefficient(df, variable, threshold=100, includer_column='energy_100g'):
        """
        Calculer le coefficient linéaire 'a' pour une variable par rapport à l'énergie.
        
        :param df: Le DataFrame à utiliser.
        :param variable: La variable à considérer.
        :param threshold: Le seuil pour déterminer les valeurs élevées de la variable (défaut : 95).
        :param includer_column: Le nom de la colonne d'énergie (défaut : 'includer_100g').
        :return: Le coefficient linéaire 'a'. Retourne 0 si aucune ligne avec une valeur élevée pour la variable n'est trouvée.
        """
        
        # Filtrer le DataFrame pour ne garder que les lignes où la variable est supérieure à threshold
        high_variable_df = df[df[variable] >= threshold]
        
        # Vérifier que high_variable_df n'est pas vide
        if high_variable_df.empty:
            # Si c'est vide, retourner 0
            return 0
        else:
            # Sinon, calculer le coefficient
            mean_includer = high_variable_df[includer_column].mean()
            a = mean_includer / threshold
            return a

    def get_includer_coefficient(df, variable, threshold=100, includer_column='energy_100g'):
        """
        Calculer le coefficient linéaire 'a' pour une variable par rapport à l'énergie.
        
        :param df: Le DataFrame à utiliser.
        :param variable: La variable à considérer.
        :param threshold: Le seuil pour déterminer les valeurs élevées de la variable (défaut : 95).
        :param includer_column: Le nom de la colonne d'énergie (défaut : 'includer_100g').
        :return: Le coefficient linéaire 'a'. Retourne 0 si aucune ligne avec une valeur élevée pour la variable n'est trouvée.
        """
        
        # Filtrer le DataFrame pour ne garder que les lignes où la variable est supérieure à threshold
        high_variable_df = df[df[variable] >= threshold]
        
        # Vérifier que high_variable_df n'est pas vide
        if high_variable_df.empty:
            # Si c'est vide, retourner 0
            return 0
        else:
            # Sinon, calculer le coefficient
            mean_includer = high_variable_df[includer_column].mean()
            a = mean_includer / threshold
            
            # Remplacer les valeurs de 'includer_column' par NaN lorsque 'variable' est égal à 100 et 'includer_column' est supérieure à a*100
            df.loc[(df[variable] == 100) & (df[includer_column] > a*100), includer_column] = np.nan
            
            return a



    def clean_variable_by_inclusion_constraint(df, a, variable, includer_column='energy_100g'):
        """
        Nettoyer le DataFrame en supprimant les lignes qui ne respectent pas la contrainte de l'énergie.
    
        :param df: Le DataFrame à nettoyer.
        :param a: Le coefficient linéaire pour la variable.
        :param variable: Le nom de la variable.
        :param includer_column: Le nom de la colonne d'énergie (par défaut : 'includer_100g').
        :return: Le DataFrame nettoyé.
        """
        # Calculez l'énergie minimale que chaque produit doit avoir
        df['min_includer'] = a * df[variable]
    
        # Identifiez les lignes qui ne respectent pas la contrainte
        df['violation'] = np.where(df[includer_column] < df['min_includer'], 1, 0)
    
        # Supprimez les lignes qui ne respectent pas la contrainte
        df_clean = df[df['violation'] == 0]
    
        # Supprimez les colonnes temporaires
        df_clean = df_clean.drop(['min_includer', 'violation'], axis=1)
    
        return df_clean
    
    
    for variable in involved_variables:
        a = get_includer_coefficient(df, variable, includer_column=includer_column)
        df = clean_variable_by_inclusion_constraint(df, a, variable, includer_column)
    return df


# In[4]:


def filter_data_by_missing_rate_threshold(dataframe, missing_rate_threshold):
    """
    Cette fonction prend en entrée un DataFrame Pandas et un seuil de taux d'absence,
    elle renvoie un DataFrame nettoyé en supprimant les variables avec un taux d'absence supérieur au seuil.
    """
    # Calcul du pourcentage de données manquantes pour chaque colonne
    missing_data = dataframe.isnull().mean() * 100

    # Filtrer les colonnes avec un taux d'absence inférieur ou égal au seuil
    columns_to_keep = missing_data[missing_data <= missing_rate_threshold].index

    # Supprimer les colonnes avec un taux d'absence supérieur au seuil
    cleaned_dataframe = dataframe[columns_to_keep]

    return cleaned_dataframe


# In[ ]:





# In[5]:


def clean_dataframe_by_inclusion_constraints(df, inclusion_constraints):
    df = df.copy()
    for v1, v2 in inclusion_constraints:
        mask = df[v1] > df[v2]
        df.loc[mask, [v1, v2]] = np.nan
    return df



# In[6]:


def clean_dataframe_by_constraints(df):
    """
    Nettoie un DataFrame en appliquant plusieurs contraintes.

    :param df: DataFrame à nettoyer.
    :return: DataFrame nettoyé.
    """
    # Colonnes de valeurs pouvant sans aberration sortir de l'intervalle [0;100]
    excluded_columns=['energy_100g','nutrition-score-fr_100g','nutrition-score-uk_100g','code','nutrition_grade_fr']

    # Suppression dans numeric_data des valeurs impossibles dans les colonnes necessairement entre 0 et 100
    # Nettoyage de la data par application des contraintes de domaine
    df = clean_dataframe_by_domain(df, excluded_columns)

    # Contraintes d'inclusion
    inclusion_constraints = [('saturated-fat_100g', 'fat_100g'), 
                             ('sugars_100g', 'carbohydrates_100g'),
                             ('trans-fat_100g','fat_100g'),
                             ('sodium_100g','salt_100g')]

    # Nettoyage de la data par application des contraintes d'inclusion
    df = clean_dataframe_by_inclusion_constraints(df, inclusion_constraints)

    # Contraintes de participation à 'energy_100'
    involved_variables=['carbohydrates_100g','fat_100g','saturated-fat_100g','sugars_100g','trans-fat_100g']

    # Nettoyage de la data par application des contraintes de participation
    df = clean_variables_by_participation_constraints(df, involved_variables, includer_column='energy_100g')

    # Contraintes de participation à 'carbohydrates_100g'
    involved_variables=['sugars_100g']

    # Nettoyage de la data par application des contraintes de participation
    df = clean_variables_by_participation_constraints(df, involved_variables, includer_column='carbohydrates_100g')

    return df


# In[ ]:





# ### Fonctions de Visualisation des Données En :

# #### Exploration

# In[7]:


import pandas as pd
import numpy as np

def columns_with_presence_percentage(data, threshold):
    # Calculer le pourcentage de présence de données pour chaque variable
    presence_percentage = (1 - data.isnull().sum() / len(data)) * 100

    # Obtenir le type de données de chaque colonne dans le DataFrame original
    data_types = data.dtypes

    # Calculer la variance uniquement pour les colonnes numériques
    variances = data.select_dtypes(include=[np.number]).var()

    # Créer un nouveau DataFrame avec les noms de variables et leur pourcentage de présence correspondant
    columns_with_presence_percentage = pd.DataFrame({'Variable': presence_percentage.index, 'PresencePercentage': presence_percentage.values})

    # Ajouter une nouvelle colonne au DataFrame qui contient la variance de chaque variable
    columns_with_presence_percentage['Variance'] = columns_with_presence_percentage['Variable'].map(variances)

    # Filtrer le DataFrame pour n'inclure que les colonnes dont le pourcentage de présence est supérieur au seuil spécifié
    columns_with_presence_percentage = columns_with_presence_percentage[columns_with_presence_percentage['PresencePercentage'] > threshold]

    # Ajouter une nouvelle colonne au DataFrame qui contient le type de données de chaque variable
    columns_with_presence_percentage['Type'] = columns_with_presence_percentage['Variable'].map(lambda var: str(data_types[var]))

    # Trier le DataFrame par pourcentage de présence en ordre décroissant, et par nom de variable en ordre croissant en cas d'égalité
    columns_with_presence_percentage = columns_with_presence_percentage.sort_values(by=['PresencePercentage', 'Variable'], ascending=[False, True])

    # Formater le pourcentage de présence pour n'avoir que deux chiffres après la virgule
    columns_with_presence_percentage['PresencePercentage'] = columns_with_presence_percentage['PresencePercentage'].map('{:.2f}'.format)

    # Réinitialiser l'index du DataFrame après le tri
    columns_with_presence_percentage = columns_with_presence_percentage.reset_index(drop=True)

    # Renvoyer le DataFrame final
    return columns_with_presence_percentage



# #### Aberration

# In[8]:


def plot_boxplot(dataframe):
    """
    Crée une boîte à moustaches pour chaque variable numérique du DataFrame.
    Affiche le taux d'absence dans la légende de chaque boîte à moustaches.

    :param dataframe: DataFrame numérique
    """
    # Parcours des colonnes du DataFrame
    for column in dataframe.columns:
        # Vérifier si la colonne est numérique
        if np.issubdtype(dataframe[column].dtype, np.number):
            # Calcul du taux d'absence pour la colonne courante
            missing_rate = dataframe[column].isnull().mean() * 100

            # Création du boxplot pour la colonne courante
            plt.figure()
            dataframe[column].plot(kind='box')
            plt.title(f'Boîte à moustaches pour {column}')
            plt.ylabel('Valeurs')

            # Calcul des statistiques remarquables
            median = dataframe[column].median()
            q1 = dataframe[column].quantile(0.25)
            q3 = dataframe[column].quantile(0.75)
            lower_fence = q1 - 1.5 * (q3 - q1)
            upper_fence = q3 + 1.5 * (q3 - q1)
            # Recherche des valeurs aberrantes
            outliers = dataframe[(dataframe[column] < lower_fence) | (dataframe[column] > upper_fence)][column]
            num_outliers = outliers.shape[0]

            # Affichage des statistiques remarquables et du taux d'absence sur la figure
            plt.text(0.95, 0.90, f'Mediane: {median:.2f}', transform=plt.gca().transAxes)
            plt.text(0.95, 0.85, f'Q1: {q1:.2f}', transform=plt.gca().transAxes)
            plt.text(0.95, 0.80, f'Q3: {q3:.2f}', transform=plt.gca().transAxes)
            plt.text(0.95, 0.75, f'Limite inférieure: {lower_fence:.2f}', transform=plt.gca().transAxes)
            plt.text(0.95, 0.70, f'Limite supérieure: {upper_fence:.2f}', transform=plt.gca().transAxes)
            plt.text(0.95, 0.05, f'Taux d\'absence: {missing_rate:.2f}%', transform=plt.gca().transAxes)
            plt.text(0.95, 0.55, f'Nombre de valeurs aberrantes: {num_outliers}', transform=plt.gca().transAxes)

            plt.show()


# #### Analyse Univariée

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns

def compare_variable_density_before_after_imputation(df_before, df_after, column, method):
    plt.figure(figsize=(12, 6))

    sns.kdeplot(df_before[column], color='blue', label='Avant imputation')
    sns.kdeplot(df_after[column], color='red', label=f'Après imputation ({method})')

    plt.title(f"Distribution de la variable {column} avant et après imputation avec {method}")
    plt.legend()
    plt.show()

def compare_densities_before_after_imputation(df_before, df_after, method):
    df_after= df_after[df_before.columns]
    # S'assurer que les deux dataframes sont alignées sur les mêmes colonnes
    assert df_before.columns.equals(df_after.columns)

    # Pour chaque colonne dans le dataframe
    for column in df_before.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df_before[column]):
            # Affiche le graphique de densité avant et après imputation pour cette colonne
            compare_variable_density_before_after_imputation(df_before, df_after, column, method)




# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score


def plot_high_corr_pairs_before_after_imputation(df_before, df_after, inclusion_constraints=[],correlation_threshold=0.5, imputation_methode="imputation_methode"):

    """
    Cette fonction trace des graphiques de paires de variables ayant une corrélation supérieure à un certain seuil dans le DataFrame après l'imputation. 
    Pour chaque paire, elle trace également un graphique bivarié avant et après imputation.

    """
    
    # Fonction pour obtenir des paires de caractéristiques fortement corrélées
    def get_high_corr_pairs(corr_matrix, correlation_threshold):
        pairs = []
        # Boucle sur la matrice de corrélation
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                # Si la corrélation absolue est supérieure au seuil, ajouter la paire
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        return pairs
    
    # Ensemble pour stocker les paires qui ont déjà été tracées
    plotted_pairs = set()

    # Calcule la matrice de corrélation uniquement pour les variables numériques
    df_after_numeric = df_after.select_dtypes(exclude=['object'])
    corr_matrix = df_after_numeric.corr()

    # Obtient les paires de caractéristiques fortement corrélées
    high_corr_pairs = get_high_corr_pairs(corr_matrix, correlation_threshold=correlation_threshold)

    # Aligner les indices de df_before et df_after
    df_before = df_before.reindex(df_after.index)

   # Boucle sur les paires de caractéristiques fortement corrélées
    for col1, col2 in high_corr_pairs:
        # Vérifie si la paire ou son inverse a déjà été tracée
        if (col1, col2) not in plotted_pairs and (col2, col1) not in plotted_pairs:
            # Effectue une analyse bivariée et trace le graphique
            bivariate_analysis(df_before, df_after, var1=col1, var2=col2,inclusion_constraints=inclusion_constraints, degree=1, imputation_methode=imputation_methode)
            # Ajoute la paire à l'ensemble des paires déjà tracées
            plotted_pairs.add((col1, col2))


# In[11]:


import matplotlib.pyplot as plt

def compare_histograms_before_after_occurrences(dataframe_before, dataframe_after):
    """
    Affiche itérativement les histogrammes des variables spécifiées dans une liste.

    Parameters:
        df_before (pandas DataFrame): Le DataFrame contenant les données avant transformation.
        df_after (pandas DataFrame): Le DataFrame contenant les données après transformation.

    Returns:
        None (Affiche les histogrammes directement).
    """
    df_before=dataframe_before.copy()
    df_after=dataframe_after.copy()
    # Suppression de la colonne 'code' si elle existe
    if 'code' in df_before.columns:
        df_before = df_before.drop('code', axis=1)
    if 'code' in df_after.columns:
        df_after = df_after.drop('code', axis=1)

    # Obtenir la liste des variables à partir des colonnes de df_before
    variables_list = df_before.columns

    # Parcours des variables dans la liste
    for variable in variables_list:
        # Vérification si la variable est présente dans les DataFrame
        if variable in df_after.columns:
            # Calcul du min et du max pour l'intervalle d'affichage
            min_value = min(df_before[variable].min(), df_after[variable].min())
            max_value = max(df_before[variable].max(), df_after[variable].max())

            # Création de l'histogramme pour la variable avec l'intervalle spécifié
            plt.hist(df_before[variable], bins=20, range=(min_value - 1, max_value + 1), alpha=0.5, label='Avant', color='blue')
            plt.hist(df_after[variable], bins=20, range=(min_value - 1, max_value + 1), alpha=0.5, label='Après', color='red')
            
            plt.xlabel(variable)
            plt.ylabel('Nombre d\'occurrences')
            plt.title(f'Histogramme de {variable}')
            plt.legend(loc='upper right')
            plt.show()
        else:
            print(f'La variable "{variable}" n\'existe pas dans les DataFrame.')



# In[12]:


import matplotlib.pyplot as plt

def compare_histograms_before_after_frequencies(dataframe_before, dataframe_after):
    """
    Affiche itérativement les histogrammes des variables spécifiées dans une liste.

    Parameters:
        df_before (pandas DataFrame): Le DataFrame contenant les données avant transformation.
        df_after (pandas DataFrame): Le DataFrame contenant les données après transformation.

    Returns:
        None (Affiche les histogrammes directement).
    """
    df_before=dataframe_before.copy()
    df_after=dataframe_after.copy()
    # Suppression de la colonne 'code' si elle existe
    if 'code' in df_before.columns:
        df_before = df_before.drop('code', axis=1)
    if 'code' in df_after.columns:
        df_after = df_after.drop('code', axis=1)

    # Obtenir la liste des variables à partir des colonnes de df_before
    variables_list = df_before.columns

    # Parcours des variables dans la liste
    for variable in variables_list:
        # Vérification si la variable est présente dans les DataFrame
        if variable in df_after.columns:
            # Calcul du min et du max pour l'intervalle d'affichage
            min_value = min(df_before[variable].min(), df_after[variable].min())
            max_value = max(df_before[variable].max(), df_after[variable].max())

            # Création de l'histogramme pour la variable avec l'intervalle spécifié
            plt.hist(df_before[variable], bins=20, range=(min_value - 1, max_value + 1), alpha=0.5, label='Avant', color='blue', density=True)
            plt.hist(df_after[variable], bins=20, range=(min_value - 1, max_value + 1), alpha=0.5, label='Après', color='red', density=True)
            
            plt.xlabel(variable)
            plt.ylabel('Fréquence')
            plt.title(f'Histogramme de {variable}')
            plt.legend(loc='upper right')
            plt.show()
        else:
            print(f'La variable "{variable}" n\'existe pas dans les DataFrame.')


# #### Analyse Bivariée

# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_high_corr_pairs(df, correlation_threshold):
    """
    Trace des graphiques de dispersion pour toutes les paires de colonnes qui ont une corrélation supérieure à 
    une valeur seuil spécifiée. Trace toutes les paires, y compris les paires inversées.
    
    :param dataframe: DataFrame contenant les données.
    :param correlation_threshold: Valeur seuil de corrélation pour déterminer quelles paires tracer.
    """
    
    # Sélectionner uniquement les colonnes numériques
    dataframe = df.select_dtypes(exclude=['object'])
    
    # Calculer la matrice de corrélation
    corr_matrix = dataframe.corr()

    # Obtenir les paires de colonnes avec une corrélation supérieure à correlation_threshold
    high_corr_pairs = [(col1, col2) for col1 in corr_matrix.columns for col2 in corr_matrix.columns if
                       (corr_matrix.loc[col1, col2] > correlation_threshold) and (col1 != col2)]

    # Créer un ensemble pour stocker les colonnes déjà tracées
    plotted_pairs = set()

    # Créer un graphique de dispersion pour chaque paire
    for pair in high_corr_pairs:
        col1, col2 = pair
        # Vérifier si la paire ou sa version inversée a déjà été tracée
        if (col1, col2) not in plotted_pairs and (col2, col1) not in plotted_pairs:
            sns.scatterplot(data=dataframe, x=col1, y=col2)
            plt.xlabel(col1)
            plt.ylabel(col2)
            plt.title(f"Cof de correlation : {corr_matrix.loc[col1, col2]:.2f} ")
            plt.show()
        plotted_pairs.add((col1, col2))  # Ajouter la paire à l'ensemble des paires tracées



# In[14]:


def bivariate_analysis(df_before, df_after, var1, var2, degree, inclusion_constraints=[],imputation_methode="imputation_methode"):
    """
    Cette fonction effectue une analyse bivariée, c'est-à-dire qu'elle analyse la relation entre deux variables. 
    Elle commence par créer un masque pour les valeurs non NaN dans les deux variables, 
    puis elle crée un sous-ensemble de df_before et df_after où les deux variables ne sont pas NaN. 
    Ensuite, elle crée un sous-ensemble du dataframe imputé et calcule la corrélation entre les deux variables.

    La fonction trace ensuite un graphique de dispersion, avec les points non imputés en bleu et les points imputés en rouge. 
    Elle effectue ensuite un ajustement polynomial et trace la ligne d'ajustement. Enfin, elle affiche le graphique.

    """

    # Aligner les indices de df_before et df_after
    df_before, df_after = df_before.align(df_after, join='inner', axis=0)
    
    # Créer un masque pour les valeurs non NaN dans les deux variables
    mask = df_after[var1].notna() & df_after[var2].notna()
    
    # Créer un sous-ensemble de df_before et df_after où les deux variables ne sont pas NaN
    subset_after = df_after.loc[mask, [var1, var2]]
    subset_before = df_before.loc[mask, [var1, var2]]


    # Créer un sous-ensemble du dataframe imputé
    imputed_subset = (subset_before.isna() & subset_after.notna())

    # Calculer la corrélation
    corr = subset_after[var1].corr(subset_after[var2])
    print(f"Corrélation entre {var1} et {var2}: {corr}")
    
    # Afficher un graphique de dispersion
    plt.figure(figsize=(6, 6))
    # Tracer les points non imputés
    sns.scatterplot(data=subset_after[~imputed_subset[var1] & ~imputed_subset[var2]], x=var1, y=var2, color='blue', label='Non imputés')
    # Tracer les points imputés
    sns.scatterplot(data=subset_after[imputed_subset[var1] | imputed_subset[var2]], x=var1, y=var2, color='red', label=f'{imputation_methode} - Imputés')

    # Ajustement polynomial
    x = subset_after[var1].values
    y = subset_after[var2].values
    coeffs = np.polyfit(x, y, degree)
    y_fit = np.polyval(coeffs, x)
    
    # Tracer la ligne d'ajustement
    plt.plot(x, y_fit, color='green')
    plt.title(f"{var1} vs {var2} (R2={r2_score(y, y_fit):.2f})")

    # Ajout de la droite y=x pour les paires spécifiques de variables
    if (var1, var2) in inclusion_constraints or (var2, var1) in inclusion_constraints:
        plt.plot([0, 100], [0, 100], color='purple', linestyle='--')
    
    # Création de la légende avec couleur personnalisée
    leg = plt.legend()
    for text, color in zip(leg.get_texts(), ['blue', 'red']):
        text.set_color(color)

    # Modifier les limites des axes pour les colonnes spécifiques
    if var1 not in ['energy_100g','nutrition-score-fr_100g','nutrition-score-uk_100g'] and var2 not in ['energy_100g','nutrition-score-fr_100g','nutrition-score-uk_100g']:
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score

def plot_high_corr_pairs_before_after_imputation(df_before, df_after, inclusion_constraints=[],correlation_threshold=0.5, imputation_methode="imputation_methode"):

    """
    Cette fonction trace des graphiques de paires de variables ayant une corrélation supérieure à un certain seuil dans le DataFrame après l'imputation. 
    Pour chaque paire, elle trace également un graphique bivarié avant et après imputation.

    """
    
    # Fonction pour obtenir des paires de caractéristiques fortement corrélées
    def get_high_corr_pairs(corr_matrix, correlation_threshold):
        pairs = []
        # Boucle sur la matrice de corrélation
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                # Si la corrélation absolue est supérieure au seuil, ajouter la paire
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
        return pairs
    
    # Ensemble pour stocker les paires qui ont déjà été tracées
    plotted_pairs = set()

    # Calcule la matrice de corrélation uniquement pour les variables numériques
    df_after_numeric = df_after.select_dtypes(exclude=['object'])
    corr_matrix = df_after_numeric.corr()

    # Obtient les paires de caractéristiques fortement corrélées
    high_corr_pairs = get_high_corr_pairs(corr_matrix, correlation_threshold=correlation_threshold)

    # Aligner les indices de df_before et df_after
    df_before = df_before.reindex(df_after.index)

   # Boucle sur les paires de caractéristiques fortement corrélées
    for col1, col2 in high_corr_pairs:
        # Vérifie si la paire ou son inverse a déjà été tracée
        if (col1, col2) not in plotted_pairs and (col2, col1) not in plotted_pairs:
            # Effectue une analyse bivariée et trace le graphique
            bivariate_analysis(df_before, df_after, var1=col1, var2=col2,inclusion_constraints=inclusion_constraints, degree=1, imputation_methode=imputation_methode)
            # Ajoute la paire à l'ensemble des paires déjà tracées
            plotted_pairs.add((col1, col2))


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_density_before_after(df_before, df_after, column, imputation_method):
    plt.figure(figsize=(12, 6))

    sns.kdeplot(df_before[column], color='blue', label='Avant imputation')
    sns.kdeplot(df_after[column], color='red', label=f'Après imputation ({imputation_method})')

    plt.title(f"Distribution de la variable {column} avant et après imputation avec {imputation_method}")
    plt.legend()
    plt.show()

def compare_density_before_after_imputation(df_before, df_after, imputation_method):
    df_before_copy=df_before.copy()
    df_after_copy=df_after.copy()
    df_after_copy= df_after_copy[df_before_copy.columns]
    
    # S'assurer que les deux dataframes sont alignées sur les mêmes colonnes
    assert df_before_copy.columns.equals(df_after_copy.columns)

    # Pour chaque colonne dans le dataframe
    for column in df_before_copy.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df_before_copy[column]):
            # Affiche le graphique de densité avant et après imputation pour cette colonne
            plot_density_before_after(df_before_copy, df_after, column, imputation_method)




# #### Cercle de corrélation

# In[16]:


from adjustText import adjust_text
def plot_correlation_circle(data, components=[0,1], circle_radius=1, cols_to_drop=None):
    """
    Fonction pour tracer le cercle des corrélations de l'ACP.

    :param data: DataFrame contenant les données d'origine.
    :param components: Les composantes principales à utiliser pour le tracé. Par défaut, [0,1].
    :param circle_radius: Rayon du cercle à tracer. Par défaut, 0.8.
    :param cols_to_drop: Liste de colonnes à supprimer avant d'exécuter la PCA. Par défaut, None.
    """
    # Supprimez les colonnes indiquées du DataFrame avant de passer à la PCA
    if cols_to_drop is not None:
        data = data.drop(cols_to_drop, axis=1)

    pca = PCA(n_components=max(components)+1) # assurez-vous d'avoir calculé suffisamment de composantes
    pca.fit(data)
    pcs = pca.components_

    fig, ax = plt.subplots(figsize=(12,12))

    for i in range(pcs.shape[1]):
        x, y = pcs[components[0], i], pcs[components[1], i]
        
        # déterminez la couleur en fonction de la corrélation avec l'axe
        if x > 0 and y > 0:
            color = 'red'
        elif x > 0 and y < 0:
            color = 'blue'
        elif x < 0 and y > 0:
            color = 'green'
        else:
            color = 'black'
        
        plt.quiver(0, 0, x, y, width=0.0025, angles='xy', scale_units='xy', scale=1, color=color)

    circle = plt.Circle((0,0), circle_radius, facecolor='none', edgecolor='olive')
    ax.add_artist(circle)

    plt.plot([-circle_radius, circle_radius], [0, 0], color='black', ls='--')
    plt.plot([0, 0], [-circle_radius, circle_radius], color='black', ls='--')  

    texts = []
    for i, (x, y) in enumerate(zip(pcs[components[0]], pcs[components[1]])):
        texts.append(plt.text(x, y, data.columns[i], fontsize='8', ha='center'))

    adjust_text(texts)

    plt.xlim(-circle_radius, circle_radius)
    plt.ylim(-circle_radius, circle_radius)

    plt.show()


# ### Fonctions d'Imputation

# In[70]:


from sklearn.impute import SimpleImputer

def impute_knn(dataframe, categorical_columns=[], n_neighbors=5):
    """
    Cette fonction effectue une imputation k-NN (k-nearest neighbors) sur un DataFrame.
    """

    # Enlève les colonnes qui sont entièrement nulles
    dataframe = dataframe.dropna(axis=1, how='all')

    # Sauvegarde et suppression des colonnes catégorielles
    saved_columns = {}
    for column in categorical_columns:
        if column in dataframe.columns:
            saved_columns[column] = dataframe[column].copy()
            dataframe = dataframe.drop(columns=[column])

    # Crée un imputer KNN
    imputer = KNNImputer(n_neighbors=n_neighbors, weights='uniform', metric='nan_euclidean')
    # Fit l'imputer sur le DataFrame et transforme le DataFrame
    imputed_data = imputer.fit_transform(dataframe)

    # Crée un nouveau DataFrame à partir des données imputées
    imputed_df = pd.DataFrame(imputed_data, columns=dataframe.columns, index=dataframe.index)

    # Si les colonnes catégorielles étaient dans les colonnes originales et ne sont pas dans les colonnes du DataFrame imputé,
    # insère les colonnes catégorielles dans le DataFrame imputé
    for column, data in saved_columns.items():
        if column not in imputed_df.columns:
            imputed_df.insert(0, column, data)

    # Retourne le DataFrame imputé
    return imputed_df


from sklearn.impute import SimpleImputer

def impute_mean(dataframe, categorical_columns=[]):
    """
    Cette fonction effectue une imputation moyenne sur un DataFrame.
    """

    # Enlève les colonnes qui sont entièrement nulles
    dataframe = dataframe.dropna(axis=1, how='all')

    # Sauvegarde et suppression des colonnes catégorielles
    saved_columns = {}
    for column in categorical_columns:
        if column in dataframe.columns:
            saved_columns[column] = dataframe[column].copy()
            dataframe = dataframe.drop(columns=[column])

    # Crée un imputer qui utilise la stratégie de la moyenne pour remplacer les valeurs manquantes
    imputer = SimpleImputer(strategy='mean')

    # Fit l'imputer sur le DataFrame et transforme le DataFrame
    imputed_data = imputer.fit_transform(dataframe)

    # Crée un nouveau DataFrame à partir des données imputées
    imputed_df = pd.DataFrame(imputed_data, columns=dataframe.columns, index=dataframe.index)

    # Si les colonnes catégorielles étaient dans les colonnes originales et ne sont pas dans les colonnes du DataFrame imputé,
    # insère les colonnes catégorielles dans le DataFrame imputé
    for column, data in saved_columns.items():
        if column not in imputed_df.columns:
            imputed_df.insert(0, column, data)

    # Retourne le DataFrame imputé
    return imputed_df


from sklearn.impute import KNNImputer


def impute_pivot(df, pivots, impute, categorical_columns):

    """
    Cette fonction effectue une imputation des valeurs manquantes dans un DataFrame, segmenté par une liste de colonnes pivot. 
    Pour chaque segment, si toutes les autres colonnes sont nulles, elle retourne le segment tel quel sans imputation. 
    Sinon, elle utilise la fonction d'imputation donnée pour imputer les valeurs manquantes dans les colonnes non catégorielles.
    Elle retourne le DataFrame imputé et un DataFrame contenant les lignes où les pivots étaient manquants.

    """
    start_time = time.time()
    data=df.copy()

    def remove_missing_pivot(df, pivots):
        missing_pivot_df = df[df[pivots].isna().any(axis=1)]
        df = df[df[pivots].notna().all(axis=1)]
        return df, missing_pivot_df

    def segment_dataframe(df, pivots):
        segments = [group for _, group in df.groupby(pivots)]
        return segments

    def impute_segments(segments, impute):
        imputed_segments = []
        for segment in segments:
            if segment.drop(columns=categorical_columns).isnull().all().all():
                imputed_segments.append(segment)
            else:
                imputed_segments.append(impute(segment.drop(columns=categorical_columns)))
        # Concatenate the imputed segments before returning them
        imputed_df = pd.concat(imputed_segments)
        return imputed_df, missing_pivot_df


    df, missing_pivot_df = remove_missing_pivot(df, pivots)
    segments = segment_dataframe(df, pivots)
    sorted_segments = sorted(segments, key=lambda x: tuple(x[pivot].iloc[0] for pivot in pivots), reverse=True)
    imputed_df, missing_pivot_df = impute_segments(sorted_segments, impute)
    # Merge the imputed dataframe with categorical columns
    imputed_df = pd.concat([imputed_df, df[categorical_columns]], axis=1)
    imputed_df = imputed_df.dropna()
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Temps d'exécution : {execution_time} secondes")
    
    return imputed_df, missing_pivot_df

from concurrent.futures import ThreadPoolExecutor

def impute_segments(segments, impute):
    imputed_segments = []

    def impute_segment(segment):
        if segment.drop(columns=categorical_columns).isnull().all().all():
            return segment
        else:
            return impute(segment.drop(columns=categorical_columns))

    with ThreadPoolExecutor() as executor:
        imputed_segments = list(executor.map(impute_segment, segments))

    # Concatenate the imputed segments before returning them
    imputed_df = pd.concat(imputed_segments)
    return imputed_df



# ### Fonctions PCA

# In[18]:


from sklearn.preprocessing import StandardScaler

def center_and_normalize(df):
    scaler = StandardScaler()
    
    # Vérifiez si 'code' est dans les colonnes de df
    codes = df['code'] if 'code' in df.columns else None
    
    # Si 'code' est dans les colonnes, créez une copie de df sans 'code'
    if codes is not None:
        df_to_scale = df.drop('code', axis=1)
    else:
        df_to_scale = df.copy()
    
    # Mettez à l'échelle le DataFrame sans la colonne 'code'
    scaled_values = scaler.fit_transform(df_to_scale)
    
    # Remettre les données mises à l'échelle dans un DataFrame
    scaled_df = pd.DataFrame(scaled_values, columns=df_to_scale.columns)
    
    # Si 'code' était dans les colonnes, rajoutez-le
    if codes is not None:
        scaled_df = pd.concat([codes, scaled_df], axis=1)

    return scaled_df



def inverse_center_and_normalize(df, scaler):
    # Vérifiez si 'code' est dans les colonnes de df
    codes = df['code'] if 'code' in df.columns else None

    # Si 'code' est dans les colonnes, créez une copie de df sans 'code'
    if codes is not None:
        df_to_inverse = df.drop('code', axis=1)
    else:
        df_to_inverse = df.copy()

    # Inversez la mise à l'échelle sur le DataFrame sans la colonne 'code'
    inversed_values = scaler.inverse_transform(df_to_inverse)

    # Remettez les données inversées dans un DataFrame
    inversed_df = pd.DataFrame(inversed_values, columns=df_to_inverse.columns)

    # Si 'code' était dans les colonnes, rajoutez-le
    if codes is not None:
        inversed_df = pd.concat([codes, inversed_df], axis=1)

    return inversed_df


# In[19]:


from sklearn.decomposition import PCA
import pandas as pd

def perform_pca(df, n_components, categorical_column):
    # Séparer les données numériques et catégorielles
    numeric_data = df.drop(categorical_column, axis=1)
    categorical_data = df[categorical_column]
    
    # Effectuer l'analyse PCA sur les données numériques
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(numeric_data)
    
    # Créer un DataFrame avec les résultats de la PCA
    pca_df = pd.DataFrame(data = pca_data, columns = ['PC'+str(i+1) for i in range(n_components)])
    
    # Réassembler les données
    pca_df_with_categorical = pd.concat([categorical_data.reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    return pca_df_with_categorical, pca_df


# In[20]:


def perform_pca(df, n_components, categorical_column):
    # Séparer les données numériques et catégorielles
    numeric_data = df.drop(categorical_column, axis=1)
    categorical_data = df[categorical_column]
    
    # Supprimez 'index' et 'code' du DataFrame numérique s'ils existent
    if 'index' in numeric_data.columns:
        numeric_data = numeric_data.drop('index', axis=1)
    if 'code' in numeric_data.columns:
        numeric_data = numeric_data.drop('code', axis=1)

    # Effectuer l'analyse PCA sur les données numériques
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(numeric_data)
    
    # Créer un DataFrame avec les résultats de la PCA
    pca_df = pd.DataFrame(data = pca_data, columns = ['PC'+str(i) for i in range(n_components)])
    
    # Réassembler les données avec 'code'
    pca_df_with_categorical = pd.concat([df[categorical_column].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    return pca_df_with_categorical, pca

def find_variable_alignment_with_axes(pca, df, threshold=0.98):
    component_matrix = pca.components_
    variable_alignment = {name: [] for name in df.columns}

    for i, component in enumerate(component_matrix):
        # Calculer le carré des poids des variables
        squares = component**2
        # Trouver les variables pour lesquelles le poids carré est supérieur au seuil
        significant_variables = [index for index, weight in enumerate(squares) if weight > threshold]
        # Si la liste n'est pas vide, ajouter au dictionnaire
        if significant_variables:
            # Convertir les index en noms de variables en utilisant le nom des colonnes d'origine
            for index in significant_variables:
                variable_name = df.columns[index]
                variable_alignment[variable_name].append(f"PC{i}")

    return variable_alignment



# In[21]:


def perform_pca_and_alignment(df, n_components, cols_to_drop,threshold=0.99):
    # Effectuez le PCA et la recherche d'alignement
    pca_df_with_categorical, pca = perform_pca(df, n_components, 'code')

    # Supprimez les colonnes indiquées du DataFrame avant de passer à la fonction find_variable_alignment_with_axes
    numeric_data_for_alignment = df.drop(cols_to_drop, errors='ignore', axis=1)

    variable_alignment_dict = find_variable_alignment_with_axes(pca, numeric_data_for_alignment,threshold)

    # Réorganisez le dictionnaire par ordre croissant de composantes principales
    variable_alignment_dict = {k: v for k, v in sorted(variable_alignment_dict.items(), key=lambda item: item[1][0] if item[1] else '')}

    # Affichez le dictionnaire une seule fois après la réorganisation
    for variable, pcs in variable_alignment_dict.items():
        print(f"{variable}: {', '.join(pcs)}")
    
    return pca_df_with_categorical, variable_alignment_dict,pca


# ### Fonctions d'Anova

# In[22]:


def add_categorical_to_imputed(imputed, data, cat_var):
    """
    Cette fonction intègre une variable catégorielle spécifique de 'data' à 'imputed' sur la base de la colonne 'code'.
    
    :param imputed: DataFrame imputé
    :param data: DataFrame original
    :param cat_var: Nom de la colonne catégorielle à ajouter
    :return: DataFrame avec variable catégorielle intégrée
    """
    # Créez un DataFrame pour la variable catégorielle
    cat_df = data[['code', cat_var]]

    # Assurez-vous que 'code' est une colonne dans cat_df
    assert 'code' in cat_df.columns

    # Réglez 'code' comme index pour cat_df
    cat_df.set_index('code', inplace=True)

    # Tronquez 'cat_df' pour qu'il ne contienne que les lignes qui sont également dans 'imputed'
    cat_df = cat_df.loc[imputed.index]

    # Assurez-vous que 'code' est l'index pour les deux DataFrames
    assert imputed.index.name == 'code'
    assert cat_df.index.name == 'code'

    # Effectuez la concaténation
    imputed = pd.concat([imputed, cat_df], axis=1)

    return imputed

def integrate_categoricals_to_imputed(imputed, data):
    """
    Cette fonction intègre toutes les variables catégorielles de 'data' à 'imputed' sur la base de la colonne 'code'.
    
    :param imputed: DataFrame imputé
    :param data: DataFrame original
    :return: DataFrame avec variables catégorielles intégrées
    """
    # Identifiez les colonnes catégorielles
    categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
    
    # Retirez 'code' de la liste des colonnes catégorielles
    if 'code' in categorical_columns:
        categorical_columns.remove('code')

    for cat_var in categorical_columns:
        imputed = add_categorical_to_imputed(imputed, data, cat_var)
        
    return imputed


# In[23]:


def merge_dataframes(df_child, df_parent, common_column):
    """
    Une fonction pour traiter deux dataframes, supprimer les lignes de df_parent qui ne sont pas dans df_child, et concaténer 
    df_child avec les colonnes manquantes de df_parent.
    
    :param df_child: DataFrame enfant, qui contient moins de colonnes que df_parent.
    :param df_parent: DataFrame parent, qui contient plus de colonnes que df_child.
    :param common_column: la colonne commune que les deux dataframes partagent.
    :return: DataFrame final après traitement.
    """
    
    # Réinitialiser les index des deux dataframes
    df_child = df_child.reset_index()
    df_parent = df_parent.reset_index()
    
    # Définir la colonne commune comme index pour les deux dataframes
    df_child.set_index(common_column, inplace=True)
    df_parent.set_index(common_column, inplace=True)
    
    # Supprimer les lignes de df_parent qui ne sont pas dans df_child
    df_parent = df_parent[df_parent.index.isin(df_child.index)]
    
    # Obtenir les colonnes qui sont dans df_parent mais pas dans df_child
    missing_cols = [col for col in df_parent.columns if col not in df_child.columns]
    
    # Extraire ces colonnes de df_parent
    missing_data = df_parent[missing_cols]
    
    # Concaténer df_child et missing_data
    df_final = pd.concat([df_child, missing_data], axis=1)
    
    # Supprimer toutes les colonnes avec au moins une valeur manquante
    df_final = df_final.dropna(how='any', axis=1)
    
    # Réinitialiser l'index
    df_final.reset_index(inplace=True)
    
    return df_final


# In[24]:


def prepare_anova_data(normalized_centered_data, cleaned_data, cat_var):
    """
    Prépare les données pour une analyse ANOVA.
    
    :param normalized_centered_data: Données numériques normalisées et centrées.
    :param cleaned_data: Données nettoyées contenant la variable catégorielle.
    :param cat_var: Nom de la variable catégorielle à réintégrer.
    
    :return: DataFrame prêt pour l'ANOVA.
    """
    
    # Vérifier si cat_var est dans cleaned_data
    if cat_var not in cleaned_data.columns:
        raise ValueError(f"'{cat_var}' n'est pas une colonne dans cleaned_data")
    
    pre_anova_data = normalized_centered_data.copy()
    # S'assurer que les deux DataFrames sont indexés de la même manière
    pre_anova_data = pre_anova_data.set_index(cleaned_data.index)

    # Réintégrer la variable catégorielle dans les données numériques
    pre_anova_data = pre_anova_data.join(cleaned_data[cat_var])

    # Supprimer les lignes où 'cat_var' est manquante
    pre_anova_data = pre_anova_data.dropna(subset=[cat_var])
    
    return pre_anova_data
    
from scipy import stats

def perform_anova(data, cat_var, cont_var):
    """
    Effectue une ANOVA à un facteur sur une variable continue pour différents groupes d'une variable catégorielle.

    :param data: DataFrame contenant les données.
    :param cat_var: Nom de la variable catégorielle.
    :param cont_var: Nom de la variable continue.
    :return: F-value et p-value de l'ANOVA.
    """
    # Obtenir les valeurs uniques de la variable catégorielle
    cat_var_unique_values = data[cat_var].unique()

    # Préparer les données pour l'ANOVA
    anova_data = [data.loc[data[cat_var]==value, cont_var] for value in cat_var_unique_values]

    # Effectuer l'ANOVA
    f_val, p_val = stats.f_oneway(*anova_data)

    return f_val, p_val

    
from concurrent.futures import ThreadPoolExecutor, as_completed
import time  
from scipy.stats import f

def perform_anova_table_parallel(normalized_and_centered_data, cleaned_data, cat_var, numerical_columns, alpha=0.05):
    """
    Effectue une analyse de l'ANOVA en parallèle pour comparer les moyennes de plusieurs groupes.
    
    :param normalized_and_centered_data: Données normalisées et centrées.
    :param cleaned_data: Données nettoyées.
    :param cat_var: Variable catégorielle pour la comparaison des groupes.
    :param numerical_columns: Colonnes numériques à comparer.
    :param alpha: Niveau de signification pour le test d'ANOVA.
    :return: DataFrame contenant les résultats de l'ANOVA.
    """
    
    def get_conclusion(p_val, f_val, crit_val):
        """
        Détermine la conclusion de l'ANOVA en fonction de la valeur p et de la valeur critique.
        
        :param p_val: Valeur p calculée lors de l'ANOVA.
        :param f_val: Valeur F calculée lors de l'ANOVA.
        :param crit_val: Valeur critique de la distribution F.
        :return: Conclusion de l'ANOVA.
        """
        if isinstance(p_val, str) and p_val.lower() == 'n/a':
            return 'No significant differences'
        elif isinstance(p_val, (int, float)) and p_val < alpha:
            return 'Significant differences'
        elif f_val > crit_val:
            return 'Significant differences'
        else:
            return 'No significant differences'
    
    start_time = time.time()
    anova_results = pd.DataFrame(columns=['CategoricalVar', 'NumericalVar', 'F-value', 'p-value'])
    pre_anova_data = prepare_anova_data(normalized_and_centered_data, cleaned_data, cat_var)
    
    if pre_anova_data[cat_var].nunique() >= 2:
        with ThreadPoolExecutor() as executor:
            futures = []
            
            for num_var in numerical_columns:
                future = executor.submit(perform_anova, pre_anova_data, cat_var, num_var)
                futures.append((cat_var, num_var, future))

            for cat_var, num_var, future in futures:
                f_val, p_val = future.result()
                crit_val = f.ppf(1 - alpha, pre_anova_data[cat_var].nunique() - 1, pre_anova_data.shape[0] - pre_anova_data[cat_var].nunique())
                new_row = pd.DataFrame({
                    'CategoricalVar': [cat_var],
                    'NumericalVar': [num_var],
                    'F-value': [f_val],
                    'p-value': [p_val]
                })
                anova_results = pd.concat([anova_results, new_row], ignore_index=True)

    conclusions = []
    for _, row in anova_results.iterrows():
        p_val = row['p-value']
        f_val = row['F-value']
        crit_val = f.ppf(1 - alpha, pre_anova_data[row['CategoricalVar']].nunique() - 1, pre_anova_data.shape[0] - pre_anova_data[row['CategoricalVar']].nunique())
        conclusions.append(get_conclusion(p_val, f_val, crit_val))
        
    anova_results['conclusion'] = conclusions
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Temps d'exécution : {execution_time} second")

from scipy.stats import f

def perform_anova_table_parallel(normalized_and_centered_data, cleaned_data, cat_var, numerical_columns, alpha=0.05):
    """
    Effectue une analyse de l'ANOVA en parallèle pour comparer les moyennes de plusieurs groupes.
    
    :param normalized_and_centered_data: Données normalisées et centrées.
    :param cleaned_data: Données nettoyées.
    :param cat_var: Variable catégorielle pour la comparaison des groupes.
    :param numerical_columns: Colonnes numériques à comparer.
    :param alpha: Niveau de signification pour le test d'ANOVA.
    :return: DataFrame contenant les résultats de l'ANOVA.
    """
    
    def get_conclusion(p_val, f_val, crit_val):
        """
        Détermine la conclusion de l'ANOVA en fonction de la valeur p et de la valeur critique.
        
        :param p_val: Valeur p calculée lors de l'ANOVA.
        :param f_val: Valeur F calculée lors de l'ANOVA.
        :param crit_val: Valeur critique de la distribution F.
        :return: Conclusion de l'ANOVA.
        """
        if isinstance(p_val, str) and p_val.lower() == 'n/a':
            return 'No significant differences'
        elif isinstance(p_val, (int, float)) and p_val < alpha:
            return 'Significant differences'
        elif f_val > crit_val:
            return 'Significant differences'
        else:
            return 'No significant differences'
    
    start_time = time.time()
    anova_results = pd.DataFrame(columns=['CategoricalVar', 'NumericalVar', 'F-value', 'p-value'])
    pre_anova_data = prepare_anova_data(normalized_and_centered_data, cleaned_data, cat_var)
    
    if pre_anova_data[cat_var].nunique() >= 2:
        with ThreadPoolExecutor() as executor:
            futures = []
            
            for num_var in numerical_columns:
                future = executor.submit(perform_anova, pre_anova_data, cat_var, num_var)
                futures.append((cat_var, num_var, future))

            for cat_var, num_var, future in futures:
                f_val, p_val = future.result()
                crit_val = f.ppf(1 - alpha, pre_anova_data[cat_var].nunique() - 1, pre_anova_data.shape[0] - pre_anova_data[cat_var].nunique())
                new_row = pd.DataFrame({
                    'CategoricalVar': [cat_var],
                    'NumericalVar': [num_var],
                    'F-value': [f_val],
                    'p-value': [p_val]
                })
                anova_results = pd.concat([anova_results, new_row], ignore_index=True)

    conclusions = []
    for _, row in anova_results.iterrows():
        p_val = row['p-value']
        f_val = row['F-value']
        crit_val = f.ppf(1 - alpha, pre_anova_data[row['CategoricalVar']].nunique() - 1, pre_anova_data.shape[0] - pre_anova_data[row['CategoricalVar']].nunique())
        conclusions.append(get_conclusion(p_val, f_val, crit_val))
    
    anova_results['conclusion'] = conclusions

    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"Temps d'exécution : {execution_time} secondes")

    return anova_results


# In[25]:


import pandas as pd
# Lecture des données
data = pd.read_csv('fr.openfoodfacts.org.products.csv', low_memory=False, sep='\t')


# In[26]:


# supprimer les lignes sans code
data = data[data['code'].notna()]
# supprimer les lignes sans 'nutrition_grade_fr'
data = data[data['nutrition_grade_fr'].notna()]


# In[27]:


from sklearn.preprocessing import OrdinalEncoder

# Création de l'objet OrdinalEncoder
oe = OrdinalEncoder(categories=[['d', 'c', 'b', 'a', 'e']]) # Ordre des catégories à définir selon votre dataset

# Appliquer la transformation à la colonne 'nutrition_grade_fr'
data['nutrition_grade_fr_encoded'] = oe.fit_transform(data[['nutrition_grade_fr']])

data['nutrition_grade_fr_encoded'].head(150)


# In[28]:


data


# In[29]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[30]:


columns_with_presence_percentage(data,-1)


# In[31]:


data.describe()


# In[32]:


pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')


# In[33]:


# Seuil d'absence de données au delà duquel une variable sera supprimée
missing_rate_threshold = 80

# Supprimer les colonnes de data qui présente plus missing_rate_threshold% de valeurs absentes
cleaned_data_by_missing_rate = filter_data_by_missing_rate_threshold(data, missing_rate_threshold)


# In[34]:


cleaned_data_by_missing_rate


# In[35]:


# Variables avec plus de missing_rate_threshold% de valeurs présentes
columns_with_presence_percentage(cleaned_data_by_missing_rate, 0)


# In[36]:


# Filtrage des données numériques uniquement
numeric_data = cleaned_data_by_missing_rate.select_dtypes(exclude=['object'])

# Inclure la colonne 'code'
numeric_data_with_code = pd.concat([numeric_data, cleaned_data_by_missing_rate['code']], axis=1)


# In[37]:


columns_with_presence_percentage(numeric_data_with_code, 0)


# In[38]:


# Nettoyer la data en lui appliquant les contraintes 
numeric_data_with_code=clean_dataframe_by_constraints(numeric_data_with_code)


# In[ ]:


plot_boxplot(numeric_data_with_code)


# In[40]:


pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')


# In[41]:


# Liste des colonnes à cibler par le traitement des aberrations
targeted_columns =['energy_100g','nutrition-score-fr_100g','nutrition-score-uk_100g']

# Suppression des outliers dans les targeted_columns par la methode des quartiles
numeric_data_with_code=replace_outliers_by_NaN(numeric_data_with_code,targeted_columns,n_iters=2)


# In[ ]:


plot_boxplot(numeric_data_with_code)


# In[43]:


numeric_data_with_code


# In[44]:


columns_with_presence_percentage(numeric_data_with_code, 0)


# In[45]:


numerical_kernel = numeric_data_with_code.dropna()


# In[46]:


numerical_kernel


# In[ ]:


# traçées des nuages de points et droites de corrélation
plot_high_corr_pairs(numerical_kernel, correlation_threshold=0.5)


# In[48]:


pca_df_with_categorical, variable_alignment_dict, pca=perform_pca_and_alignment(numerical_kernel,20,['code'])

# La fonction PCA doit déjà avoir été exécutée
explained_variance = pca.explained_variance_ratio_

# Pour afficher la part de variance expliquée par chaque composante principale
for i, exp_var in enumerate(explained_variance):
    print(f'PC{i}: {exp_var*100:.2f}%')


# In[71]:


categorical_columns=['code']
pivots=['energy_100g']


# In[72]:


imputed_numeric_data_with_code, missing_pivot_df= impute_pivot(numeric_data_with_code, pivots, impute_knn, categorical_columns)


# In[73]:


columns_with_presence_percentage(missing_pivot_df,-1)


# In[74]:


missing_pivot_df.head()


# In[75]:


# Application des contraintes sur la data imputée
imputed_numeric_data_with_code=clean_dataframe_by_constraints(imputed_numeric_data_with_code)


# In[76]:


imputed_numeric_data_with_code = imputed_numeric_data_with_code.dropna()


# In[77]:


imputed_numeric_data_with_code


# In[56]:


compare_density_before_after_imputation(numerical_kernel, imputed_numeric_data_with_code, 'Pivot_Knn')


# In[57]:


compare_histograms_before_after_occurrences(numerical_kernel, imputed_numeric_data_with_code)


# In[58]:


compare_histograms_before_after_frequencies(numerical_kernel, imputed_numeric_data_with_code)


# In[78]:


# Contraintes d'inclusion pour tracer les droites de contraintes
inclusion_constraints = [('saturated-fat_100g', 'fat_100g'), 
                         ('sugars_100g', 'carbohydrates_100g'),
                         ('trans-fat_100g','fat_100g'),
                         ('sodium_100g','salt_100g')]


# traçées des nuages de points et droites de corrélation
plot_high_corr_pairs_before_after_imputation(numeric_data_with_code, imputed_numeric_data_with_code, inclusion_constraints, correlation_threshold=0.5, imputation_methode="Pivot_Knn")


# In[60]:


"""
Il reste 198153 lignes et 20 colonnes numériques +'code', après nettoyage
"""


# In[ ]:





# In[61]:


# Desssin du cercle de corrélation :
plot_correlation_circle(imputed_numeric_data_with_code, components=[0,2], circle_radius=1, cols_to_drop=['code'])


# In[62]:


# Récupérer les poids des variables sur PC0
pc_weights = pca.components_[0]

# Créer un DataFrame pour afficher les variables et leurs poids
weights_df = pd.DataFrame({'Variable': numeric_data.columns, 'Weight on PC': pc_weights})

# Afficher le DataFrame
print(weights_df)


# In[79]:


pca_df_with_categorical, variable_alignment_dict, pca=perform_pca_and_alignment(imputed_numeric_data_with_code,20,['code'])

# La fonction PCA doit déjà avoir été exécutée
explained_variance = pca.explained_variance_ratio_

# Pour afficher la part de variance expliquée par chaque composante principale
for i, exp_var in enumerate(explained_variance):
    print(f'PC{i}: {exp_var*100:.2f}%')


# In[80]:


imputed_numeric_data_with_code = imputed_numeric_data_with_code.reset_index()
imputed_numeric_data_with_code.set_index('code', inplace=True)


# Ensuite, intégrez les variables catégorielles
imputed_with_all_categorical = integrate_categoricals_to_imputed(imputed_numeric_data_with_code, data)


# In[81]:


variables = [
    "additives_n", 
    "calcium_100g", 
    "carbohydrates_100g", 
    "cholesterol_100g", 
    "energy_100g", 
    "fat_100g", 
    "fiber_100g", 
    "ingredients_from_palm_oil_n", 
    "ingredients_that_may_be_from_palm_oil_n", 
    "iron_100g", 
    "nutrition-score-fr_100g", 
    "nutrition-score-uk_100g", 
    "proteins_100g", 
    "salt_100g", 
    "saturated-fat_100g", 
    "sodium_100g", 
    "sugars_100g", 
    "trans-fat_100g", 
    "vitamin-a_100g", 
    "vitamin-c_100g",
    
]


# In[82]:


#imputed_with_all_categorical = imputed_with_all_categorical.rename(columns={'nutrition_grade_fr_encoded': 'nutrition_grade_fr_encoded_cleaned'})

normalized_and_centered_data=center_and_normalize(imputed_numeric_data_with_code)

perform_anova_table_parallel(normalized_and_centered_data, imputed_with_all_categorical , 'nutrition_grade_fr', variables)


# In[ ]:




