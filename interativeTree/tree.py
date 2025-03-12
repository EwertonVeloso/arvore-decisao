import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import graphviz
from io import StringIO

def load_data():
    df = pd.read_csv('CarEval.csv')
    
    # Definir a ordem das categorias
    ordinal_mapping = {
        'buying': ['low', 'med', 'high', 'vhigh'],
        'maint': ['low', 'med', 'high', 'vhigh'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high']
    }
    
    # Aplicar a codificação ordinal
    ordinal_encoder = OrdinalEncoder(categories=[ordinal_mapping[col] for col in df.columns[:-1]])
    df[df.columns[:-1]] = ordinal_encoder.fit_transform(df[df.columns[:-1]])
    
    # Codificar a variável target (class values) com LabelEncoder
    label_encoder = LabelEncoder()
    df['class values'] = label_encoder.fit_transform(df['class values'])
    
    return df, label_encoder

# Função para treinar o modelo e gerar a árvore
def train_model(df):
    X = df.drop('class values', axis=1)
    y = df['class values']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, X.columns

# Função para visualizar a árvore usando Graphviz
def visualize_tree(clf, feature_names, class_names):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, feature_names=feature_names, class_names=class_names, rounded=True)
    graph = graphviz.Source(dot_data.getvalue())
    return graph

# Configuração da página do Streamlit
st.title("Visualização:")

# Carregar os dados
df, label_encoder = load_data()

# Treinar o modelo
clf, feature_names = train_model(df)

# Visualizar a árvore
graph = visualize_tree(clf, feature_names, label_encoder.classes_)
st.graphviz_chart(graph)

# Exibir informações adicionais
st.write("### Informações do Modelo")
st.write(f"Árvore de decisão da avaliação de carros")
st.write(f"Número de classes target: {len(label_encoder.classes_)}")
st.write(f"Classes: {', '.join(label_encoder.classes_)}")