Per eseguire il codice:

  - Entrare con il terminale nella cartella Progetto.
  - Per la versione sequenziale, digitare sul terminale:
  make && mpirun --oversubscribe -np 1 bin/warsim -P
  - Per la versione parallela, digitare sul terminale:
  make && mpirun --oversubscribe -np 6 bin/warsim -p -x 2 -y 3

Se si vuole eseguire in parallelo, indicare il numero di righe e colonne di processi che si vuole avere affianco a __x__ e __y__ e fare in modo che combaci con il numero di processi effettivamente usati, altrimenti va in errore la topologia.