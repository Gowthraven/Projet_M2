import json
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.colors as mcolors
import random


def interaction_matrix_global(path):
    '''Retourne la matrice d'interaction ou Aij
    correpond au nombre de fois que le neud i transitionne
    vers j avec le dictionnaire label : id'''
    with open(path) as f:
        data = json.load(f)
    label_id=dict()
    transitions= dict()
    for melodie in data:
        for i in range(len(melodie)-1):
            splitted_x =melodie[i].split('-')
            splitted_y =melodie[i+1].split('-')
            #si la note contient un tiret ou pas
            x = splitted_x[0] if len(splitted_x)==2 else "-".join(substring for substring in splitted_x[:-1])
            y= splitted_y[0] if len(splitted_y)==2 else "-".join(substring for substring in splitted_y[:-1]) 
            if x not in label_id:
                label_id[x]= len(label_id)
            if y not in label_id:
                label_id[y]= len(label_id)
            key_transition= x+" "+y
            if key_transition not in transitions:
                transitions[key_transition]=0
            transitions[key_transition]+=1
    A= np.zeros((len(label_id),len(label_id)))
    for t in transitions:
        x,y=t.split(' ')
        A[label_id[x],label_id[y]]= transitions[t]
    return A,label_id
    

def graphe_brut(A,label_id):
    '''Affiche le graphe d'interaction brut, 
    sans traitement sur l'affichage des noeuds et des arêtes '''
    G = nx.DiGraph()
    plt.figure(figsize=(10,10))
    # Ajouter des nœuds au graphe
    G.add_nodes_from(label_id)
    num_nodes=len(label_id)

    # Ajouter des arêtes pondérées au graphe en fonction de la matrice d'interaction avec poids
    for i in label_id:
        for j in label_id:
            weight = A[label_id[i],label_id[j]]
            if weight > 0:
                G.add_edge(i, j, weight=weight)
    cmap = plt.cm.get_cmap('viridis')             
    nodes_weight= [np.sum(A[:,i]) for i in range(num_nodes)]

    # Associer les couleurs les degres entrant des noeuds
    norm = mcolors.Normalize(vmin=min(nodes_weight), vmax=max(nodes_weight))
    nodes_colors = [cmap(norm(w)) for w in nodes_weight]

    pos = nx.random_layout(G)
    # Dessiner le graphe avec des arêtes pondérées
    nx.draw(G,pos, with_labels=True, node_color=nodes_colors, node_size=nodes_weight, font_size=10, font_weight='bold',width=0.2)
    # Ajouter une barre de couleur pour indiquer la correspondance poids-couleur
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Poids')
    plt.title("Graphe d'interaction pondérée avec couleurs d'arêtes")
    plt.show()


def graphe_croissant(A, label_id): 
    '''Affiche le graphe d'interaction, 
    où les noeuds sont triés par leur poids total des arêtes entrantes de façon croissante
    Est utile pour comparer avec les histogrammes.'''
    G = nx.DiGraph()
    G.add_nodes_from(label_id)
    num_nodes = len(label_id)
    plt.figure(figsize=(10,10))

    # Ajout des arêtes pondérées au graphe 
    for i in label_id:
        for j in label_id:
            weight = A[label_id[i],label_id[j]]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    # Calcul de la taille des noeuds en fonction du poids total des arêtes entrantes
    node_weights = {node: np.sum(A[:, label_id[node]]) for node in G.nodes()}
    node_sizes = [node_weights[node] for node in G.nodes()]
    
    # Trier les noeuds par leur poids
    sorted_nodes = sorted(G.nodes(), key=lambda node: node_weights[node], reverse=True)

    # Assigner des positions dans un cercle en fonction d'un ordre trié
    pos = {}
    for i, node in enumerate(sorted_nodes):
        angle = 2 * np.pi * i / num_nodes
        pos[node] = (np.cos(angle), np.sin(angle))

    # Dessiner le ghraphe 
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_sizes, cmap=plt.cm.plasma)
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Avec différentes couleurs et transparence d'arêtes
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    edge_colors = [plt.cm.plasma(weight / max(weights)) for weight in weights]
    edge_widths = [(weight / max(weights) * 5) + 0.1 for weight in weights]
    
    for (source, target, weight), color, width in zip(G.edges(data='weight'), edge_colors, edge_widths):
        alpha = 0.1 if weight < np.mean(weights) else 0.9  # Adjust transparency based on weight
        nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], width=width,
                               alpha=alpha, edge_color=[color], arrowstyle='->', arrowsize=10)
        
    # Barre de couleur pour les poids des noeuds
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(node_sizes), vmax=max(node_sizes)))
    sm.set_array([])
    plt.colorbar(sm, label='Node Weight')

    plt.title("Graphe d'interaction pondérée avec couleurs d'arêtes et ordre des noeuds")
    plt.axis('equal')  
    plt.show()


def graphe_ameliore_global(A, label_id): 
    """
    Affiche le graphe d'interaction pondérée avec des couleurs d'arêtes et des tailles de noeuds
    Avec des améliorations graphiques. """
    G = nx.DiGraph()
    G.add_nodes_from(label_id)
    num_nodes = len(label_id)
    plt.figure(figsize=(10, 10))  

    # Ajout des arêtes pondérées au graphe 
    for i in label_id:
        for j in label_id:
            weight = A[label_id[i], label_id[j]]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    # Calcul de la taille des noeuds en fonction du poids total des arêtes entrantes
    node_weights = {node: np.sum(A[:, label_id[node]]) for node in G.nodes()}
    node_sizes = [node_weights[node] for node in G.nodes()]

    # Determine le seuil d'affichage des etiquettes (exemple : 75e percentile ici)
    label_display_threshold = np.percentile(node_sizes, 75)

    # Attribuer aux noeuds des positions aléatoires dans un cercle
    # Et trier les noeuds par poids
    sorted_nodes_by_weight = sorted(G.nodes(data=True), key=lambda x: node_weights[x[0]], reverse=True)

    # Placer le plus gros noeud de maniere aléatoire, et placer les autres noeuds en verifiant les distances, pour éviter les chevauchements des noeuds interessants à afficher
    pos = {}
    angles_used = set()
    for node, data in sorted_nodes_by_weight:
        while True:
            angle = random.uniform(0, 2 * np.pi)
            if all(not np.isclose(angle, used_angle, atol=0.05) for used_angle in angles_used):  # atol dépend de la répartition que l'on souhaite 
                angles_used.add(angle)
                pos[node] = (np.cos(angle), np.sin(angle))
                break

    
    # Dessiner le ghraphe, en utilisant les positions 
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_sizes, cmap=plt.cm.plasma)

    # N'afficher les étiquettes que des noeuds selectionnés avec le seuil
    labels_to_draw = {node: node for node, size in zip(G.nodes(), node_sizes) if size >= label_display_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=8)
    
    weights = [data['weight'] for source, target, data in G.edges(data=True)]

    # Trier les arêtes par poids, pour que les arêtes de poids plus élevés soient dessinées en dernier (au dessus des autres)
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])
    

    # Avec différentes couleurs et transparence d'arêtes
    for (source, target, data) in sorted_edges:
        weight = data['weight']
        color = plt.cm.plasma(weight / max(weights))
        width = (weight / max(weights) * 5) + 0.1
        alpha = 0.1 if weight < np.mean(weights) else 0.9  # régler la transparence en fonction du poids
        nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], width=width,
                               alpha=alpha, edge_color=[color], arrowstyle='->', arrowsize=12)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(node_sizes), vmax=max(node_sizes)))
    sm.set_array([])    
    plt.title("Graphe d'interaction pondérée avec couleurs d'arêtes")
    plt.colorbar(sm, label='Poids')
    plt.axis('equal')  
    plt.show()

def process_note_simplifie(note):
    if note == "rest" or note[0] == "r":
        return "rest"
    elif len(note) > 1 and (note[1] == "#" or note[1] == "-"):
        return note[:2]
    else :
        return note[0]

def interaction_matrix_simplifie(path):
    '''Retourne la matrice d'interaction ou Aij
    correpond au nombre de fois que le neud i transitionne
    vers j avec le dictionnaire label : id

    Difference avec la version globale : On ne prend en compte 
    que les notes : A, A-, A#, B, B-, C, C#, D, D-, D#, E, E-, 
    F, F#, G, G#, rest'''
    with open(path) as f:
        data = json.load(f)

    label_id = {}
    transitions = {}
    max_label_id = -1  # Initialisation à -1 pour commencer l'indexation à 0

    for melodie in data:
        for i in range(len(melodie) - 1):
            x = process_note_simplifie(melodie[i])
            y = process_note_simplifie(melodie[i + 1])

            if x not in label_id:
                label_id[x] = len(label_id)
            if y not in label_id:
                label_id[y] = len(label_id)
            key_transition = x + " " + y
            if key_transition not in transitions:
                transitions[key_transition] = 0
            transitions[key_transition] += 1

    A = np.zeros((len(label_id), len(label_id)))
    for t in transitions:
        x, y = t.split(' ')
        A[label_id[x], label_id[y]] = transitions[t]

    return A, label_id


def graphe_ameliore_simplifie(A, label_id): 
    """
    Affiche le graphe d'interaction pondérée avec des couleurs d'arêtes et des tailles de noeuds
    Avec des améliorations graphiques particulière pour la version simplifiée avec seulement les notes. """
    G = nx.DiGraph()
    G.add_nodes_from(label_id.keys())
    plt.figure(figsize=(10, 10))

    for i in label_id:
        for j in label_id:
            weight = A[label_id[i], label_id[j]]
            if weight > 0:
                G.add_edge(i, j, weight=weight)

    node_weights = {node: np.sum(A[:, label_id[node]]) for node in G.nodes()}
    node_sizes = [node_weights[node] for node in G.nodes()]
    label_display_threshold = np.percentile(node_sizes, 20)
    sorted_nodes_by_weight = sorted(G.nodes(data=True), key=lambda x: node_weights[x[0]], reverse=True)

    pos = {}
    angles_used = set()
    for node, data in sorted_nodes_by_weight:
        while True:
            angle = random.uniform(0, 2 * np.pi)
            if all(not np.isclose(angle, used_angle, atol=0.15) for used_angle in angles_used):
                angles_used.add(angle)
                pos[node] = (np.cos(angle), np.sin(angle))
                break

    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_sizes, cmap=plt.cm.plasma)
    labels_to_draw = {node: node for node, size in zip(G.nodes(), node_sizes) if size >= label_display_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels_to_draw, font_size=8)
    weights = [data['weight'] for source, target, data in G.edges(data=True)]
    sorted_edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])

    for (source, target, data) in sorted_edges:
        weight = data['weight']
        color = plt.cm.plasma(weight / max(weights))
        width = (weight / max(weights) * 4) + 0.1
        alpha = 0.15 if weight < np.mean(weights) else 0.6
        nx.draw_networkx_edges(G, pos, edgelist=[(source, target)], width=width,
                               alpha=alpha, edge_color=[color], arrowstyle='-|>', arrowsize=10, connectionstyle='arc,rad=0.06')

    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=min(node_sizes), vmax=max(node_sizes)))
    sm.set_array([])
    plt.title("Graphe d'interaction pondérée avec couleurs d'arêtes")
    plt.colorbar(sm, label='Poids')
    plt.axis('equal')
    plt.show()
