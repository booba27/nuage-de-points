import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Configuration pour l'affichage
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 5)

def import_data(filename):
    """
    Importe les données du fichier en gérant les virgules comme séparateurs décimaux
    """
    try:
        # Lecture avec pandas en spécifiant le séparateur décimal
        df = pd.read_csv(filename, sep='\t', decimal=',')
        print("Données importées avec succès !")
        return df
    except Exception as e:
        print(f"Erreur lors de l'importation : {e}")
        return None

def calcul_plan_moindres_carres(X, Y, Z):
    """
    Calcule le plan des moindres carrés Z = a + b*X + c*Y
    """
    # Construction de la matrice A
    A = np.column_stack([np.ones(len(X)), X, Y])
    
    # Résolution par moindres carrés
    coeff, residuals, rank, s = np.linalg.lstsq(A, Z, rcond=None)
    
    a, b, c = coeff
    return a, b, c, residuals

def afficher_resultats(X, Y, Z, a, b, c, residuals):
    """
    Affiche les résultats détaillés du calcul
    """
    print("\n" + "="*60)
    print("RÉSULTATS DU PLAN DES MOINDRES CARRÉS")
    print("="*60)
    print(f"Équation du plan : Z = {a:.6f} + {b:.6f}*X + {c:.6f}*Y")
    print(f"\nCoefficients :")
    print(f"  Constante (a) = {a:.6f}")
    print(f"  Coefficient en X (b) = {b:.6f}")
    print(f"  Coefficient en Y (c) = {c:.6f}")
    
    # Calcul des valeurs prédites et résidus
    Z_pred = a + b*X + c*Y
    residus = Z - Z_pred
    
    # Statistiques d'erreur
    erreur_quadratique = np.mean(residus**2)
    std_residus = np.std(residus)
    
    print(f"\nMétriques d'erreur :")
    print(f"  Erreur quadratique moyenne : {erreur_quadratique:.6f}")
    print(f"  Écart-type des résidus : {std_residus:.6f}")
    
    # Coefficient de détermination R²
    SS_res = np.sum(residus**2)
    SS_tot = np.sum((Z - np.mean(Z))**2)
    R2 = 1 - SS_res/SS_tot
    
    print(f"  Coefficient de détermination R² : {R2:.6f}")
    print(f"  Qualité de l'ajustement : {R2*100:.2f}%")
    
    return Z_pred, residus

def afficher_tableau_comparatif(X, Y, Z, Z_pred, residus):
    """
    Affiche un tableau comparatif des valeurs réelles et prédites
    """
    print("\n" + "="*70)
    print("TABLEAU COMPARATIF : VALEURS RÉELLES vs PRÉDITES")
    print("="*70)
    print("X\tY\tZ_réel\t\tZ_prédit\tRésidu")
    print("-"*70)
    
    for i in range(len(X)):
        print(f"{X[i]:.1f}\t{Y[i]:.1f}\t{Z[i]:.8f}\t{Z_pred[i]:.8f}\t{residus[i]:.8f}")

def visualiser_resultats(X, Y, Z, Z_pred, a, b, c):
    """
    Crée les visualisations 3D du nuage de points et du plan
    """
    fig = plt.figure(figsize=(15, 6))
    
    # Sous-graphique 1 : Nuage de points et plan
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Création d'une grille pour le plan
    x_range = np.linspace(min(X), max(X), 20)
    y_range = np.linspace(min(Y), max(Y), 20)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_grid = a + b*X_grid + c*Y_grid
    
    # Tracé du plan
    surf = ax1.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.5, 
                           cmap='spring', edgecolor='none')
    
    # Tracé des points réels et prédits
    ax1.scatter(X, Y, Z, c='blue', s=50, label='Points réels', edgecolors='black')
    ax1.scatter(X, Y, Z_pred, c='red', s=50, label='Points prédits')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Nuage de points et plan des moindres carrés')
    ax1.legend()
    
    # Sous-graphique 2 : Résidus
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Tracé du plan
    ax2.plot_surface(X_grid, Y_grid, Z_grid, alpha=0.3, 
                    cmap='spring', edgecolor='none')
    
    # Tracé des points réels et prédits
    ax2.scatter(X, Y, Z, c='blue', s=50, label='Points réels', edgecolors='black')
    ax2.scatter(X, Y, Z_pred, c='red', s=50, label='Points prédits')
    
    # Tracé des lignes de résidus
    for i in range(len(X)):
        ax2.plot([X[i], X[i]], [Y[i], Y[i]], [Z[i], Z_pred[i]], 
                'r-', linewidth=2, alpha=0.7)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Visualisation des résidus')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Fonction principale
    """
    print("=== CALCUL DU PLAN DES MOINDRES CARRÉS ===")
    
    # Importation des données
    filename = 'nuage_de_points.txt'
    df = import_data(filename)
    
    if df is None:
        return
    
    # Affichage des données importées
    print("\nDonnées importées :")
    print(df.to_string(index=False, float_format='%.6f'))
    
    # Extraction des colonnes
    X = df['X'].values
    Y = df['Y'].values
    Z = df['Z'].values
    
    # Calcul du plan des moindres carrés
    a, b, c, residuals = calcul_plan_moindres_carres(X, Y, Z)
    
    # Affichage des résultats
    Z_pred, residus = afficher_resultats(X, Y, Z, a, b, c, residuals)
    
    # Tableau comparatif
    afficher_tableau_comparatif(X, Y, Z, Z_pred, residus)
    
    # Visualisation
    print("\nGénération des graphiques...")
    visualiser_resultats(X, Y, Z, Z_pred, a, b, c)

# Version alternative avec génération manuelle des données si le fichier n'existe pas
def creer_fichier_donnees():
    """
    Crée le fichier de données si il n'existe pas
    """
    data = """X	Y	Z
1	1	-2,083333333
1	2	-3,45
1	3	-4,066666667
1	4	-4,083333333
1	5	-4,9
2	1	-3,166666667
2	2	-3,633333333
2	3	-3,7
2	4	-4,516666667
2	5	-5,533333333
3	1	-2,5
3	2	-3,966666667
3	3	-4,783333333
3	4	-5,15
3	5	-6,066666667
4	1	-3,733333333
4	2	-4
4	3	-4,166666667
4	4	-5,133333333
4	5	-6,1
5	1	-3,366666667
5	2	-4,683333333
5	3	-5
5	4	-5,166666667
5	5	-6,333333333"""
    
    with open('nuage_de_points.txt', 'w', encoding='utf-8') as f:
        f.write(data)
    print("Fichier 'nuage_de_points.txt' créé avec les données fournies.")

if __name__ == "__main__":
    # Vérification si le fichier existe, sinon le créer
    try:
        with open('nuage_de_points.txt', 'r'):
            pass
    except FileNotFoundError:
        print("Création du fichier de données...")
        creer_fichier_donnees()
    
    # Exécution du programme principal
    main()