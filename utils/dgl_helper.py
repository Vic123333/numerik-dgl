"""
Hilfsfunktionen für numerische DGL-Lösung
"""
import numpy as np
import matplotlib.pyplot as plt


def euler_explizit(f, t0, y0, h, n):
    """
    Explizites Euler-Verfahren für y' = f(t, y).

    Parameter:
        f  : Funktion f(t, y)
        t0 : Startzeitpunkt
        y0 : Anfangswert y(t0)
        h  : Schrittweite
        n  : Anzahl Schritte

    Rückgabe:
        t, y : Arrays der Zeitpunkte und Näherungslösungen
    """
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0], y[0] = t0, y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(t[i], y[i])
        t[i + 1] = t[i] + h
    return t, y


def runge_kutta_4(f, t0, y0, h, n):
    """
    Klassisches Runge-Kutta-Verfahren 4. Ordnung für y' = f(t, y).
    """
    t = np.zeros(n + 1)
    y = np.zeros(n + 1)
    t[0], y[0] = t0, y0
    for i in range(n):
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t[i + 1] = t[i] + h
    return t, y


def plot_vergleich(t_num, y_num, f_exakt=None, titel="DGL-Lösung", label_num="Numerisch"):
    """
    Plottet numerische Lösung, optional gegen exakte Lösung.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t_num, y_num, "o-", markersize=3, label=label_num)
    if f_exakt is not None:
        t_fein = np.linspace(t_num[0], t_num[-1], 500)
        ax.plot(t_fein, f_exakt(t_fein), "k--", label="Exakt")
    ax.set_xlabel("t")
    ax.set_ylabel("y(t)")
    ax.set_title(titel)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig, ax
