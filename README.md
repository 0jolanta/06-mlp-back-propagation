# 06--mlp-back-propagation

Celem laboratorium jest zbudowanie sieci wielowarstowej z zastosowaniem algorytmu wstecznej propagacji błędu.

## Algorytm wstecznej propagacji błędów krok po kroku
1. warstwa wejściowa otrzymuje dane wejściowe
2. dane wejściowe są uśredniane przez wagi
3. każda warstwa ukryta przetwarza dane wyjściowe 
4. Każde wyjście jest określane jako "błąd", który jest w rzeczywistości różni się  pomiędzy rzeczywistym wyjściem a pożądanym wyjściem

## Zastosowane metryki

MSE (ang. Mean Squared Error) jest średnią kwadratową różnicy pomiędzy wartościami szacunkowymi a wartościami rzeczywistymi.

```python
  def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)
```

Druga metryka funkcja sigmoidalna, której krzywa jest w kształcie litery S. Jej środkowy fragment jest zblizony jest do linii prostej, a fragmenty skrajne
przyjmuj¡ kształt krzywej nasycenia.

```python
  def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
Pozostała tylko dokładność. Jest to jedna z metryk oceny modeli klasyfikacji. Nieformalnie dokładność to ułamek przewidywań, które nasz model
sprawdził.
    
```python
  def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()
```

