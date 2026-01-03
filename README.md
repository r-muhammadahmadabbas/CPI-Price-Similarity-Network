# Temporal and Category-Based Price Similarity Network Analysis 🇵🇰

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![NetworkX](https://img.shields.io/badge/Library-NetworkX-green)
![Pandas](https://img.shields.io/badge/Library-Pandas-orange)

## 📌 Project Overview
This project applies **Discrete Mathematical Structures**—specifically Graph Theory, Partial Orders, and Centrality Measures—to analyze the economic connectivity of **17 major Pakistani cities**. 

By utilizing monthly **Consumer Price Index (CPI)** data from the *Pakistan Bureau of Statistics (PBS)* for the years 2023, 2024, and 2025, we construct a temporal network where nodes represent cities and edges represent the similarity in their inflation behaviors.

### 🎯 Objective
To detect structural economic relationships, identifying which cities act as "price leaders" or "hubs" and how market integration evolves over time (e.g., from integration in 2024 to fragmentation in 2025).

---

**Course:** Discrete Structures  
**Instructor:** Arshad Islam  
**Semester:** 3rd

---

## ⚙️ Methodology & Mathematical Modeling

### 1. Data Preprocessing
* **Source:** Monthly CPI reports from PBS.
* **Normalization:** Z-score normalization is applied to city price vectors to focus on *relative* price changes rather than absolute costs.
* **Categories:** 51 items mapped to 7 major categories (e.g., *Food Staples*, *Utilities*, *Transport*).

### 2. Graph Construction ($G_{y,c}$)
We model the economy as a graph where:
* **Nodes ($V$):** 17 Cities (Islamabad, Karachi, Lahore, Quetta, etc.).
* **Edges ($E$):** Exist if the **Cosine Similarity** between two cities' CPI vectors exceeds a threshold $\tau$ (e.g., 0.65).

$$
CosineSim(A, B) = \frac{A \cdot B}{\|A\| \|B\|}
$$

### 3. Centrality & Weighting Schemes
To determine the "Economic Influence" of a city, we computed four centrality metrics:
* **Degree:** Direct connections.
* **Closeness:** Speed of price shock propagation.
* **Betweenness:** Role as a bridge between clusters.
* **Eigenvector:** Connection to other influential cities.

We implemented four weighting strategies to aggregate these metrics:
1. **Equal Weighting:** Baseline.
2. **Correlation-Based:** Reduces redundancy.
3. **Entropy-Based:** Highlights distinct/unique economic roles.
4. **User-Tuned (Systemic Risk):** Prioritizes "Super-Spreader" cities.

### 4. Temporal Analysis (Partial Orders)
We defined a temporal relation $T$ where year $Y$ is "included" in year $Y'$ if the network structure persists. This was visualized using **Hasse Diagrams** to show the evolution of market stability.

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python installed. Install the dependencies using:

```bash
pip install -r requirements.txt
