"""LSTM, ensamble dual de DMEL y las variantes de los experimentos."""

from torch import nn

from dmel.informer import Informer


class LSTMNet(nn.Module):
    """LSTM de 2 capas; su último estado oculto pasa por una capa lineal que emite las 48 h."""

    def __init__(self, c_in, horizon=48, hidden=256, layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):  # [B, 336, c_in] -> [B, 48]
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class TwoBranch(nn.Module):
    """Ensamble dual (Ec. 14 del paper): ŷ = rama_1(entrada_1) + rama_2(entrada_2)."""

    def __init__(self, branches):
        super().__init__()
        self.keys = [key for key, _ in branches]
        self.nets = nn.ModuleList([net for _, net in branches])

    def branch_outputs(self, inputs):
        return [net(inputs[key]) for key, net in zip(self.keys, self.nets)]

    def forward(self, inputs):
        return sum(self.branch_outputs(inputs))


# (entrada, red) de cada rama. hf/lf = RIMFs de alta/baja frecuencia del MRS; raw = caudal sin descomponer
VARIANTS = {
    "dmel": (("hf", "informer"), ("lf", "lstm")),  # propuesta del paper
    "dmel_swap": (("hf", "lstm"), ("lf", "informer")),  # ablación 1: asignación invertida (paper, sección 5.2)
    "hybrid_raw": (("raw", "informer"), ("raw", "lstm")),  # ablación 2: sin MRS (misma arquitectura)
}

NETS = {"informer": Informer, "lstm": LSTMNet}


def input_keys(name):
    """Entradas distintas que necesita una variante (p. ej. hybrid_raw usa 'raw' en ambas ramas)."""
    return sorted({key for key, _ in VARIANTS[name]})


def build_model(name, n_channels):
    """Crea la variante `name`; n_channels: dict entrada -> número de canales."""
    return TwoBranch([(key, NETS[net](n_channels[key])) for key, net in VARIANTS[name]])
