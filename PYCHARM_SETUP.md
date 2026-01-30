# PyCharm Setup Instructions

## Configure PyCharm to use Virtual Environment

Το PyCharm χρησιμοποιεί το global Python. Πρέπει να το ρυθμίσεις να χρησιμοποιεί το virtual environment.

### Method 1: Configure Interpreter (Recommended)

1. **File → Settings** (ή `Ctrl+Alt+S`)
2. **Project: Hand Sign Recognition → Python Interpreter**
3. Κάνε click στο dropdown για interpreter
4. Κάνε click **"Add Interpreter" → "Add Local Interpreter"**
5. Επίλεξε **"Existing Environment"**
6. Browse και επίλεξε: `C:\Users\xampo\Desktop\Hand Sign Recognition\.venv\Scripts\python.exe`
7. Κάνε **OK** και **Apply**

### Method 2: Quick Fix (Εάν υπάρχει ήδη interpreter)

1. Κάνε click στο Python interpreter selector (κάτω δεξιά στο PyCharm)
2. **"Add Interpreter" → "Add Local Interpreter"**
3. Επίλεξε **"Existing Environment"**
4. Browse: `.venv\Scripts\python.exe`
5. **OK**

### Method 3: Run Configuration

1. **Run → Edit Configurations**
2. Στο **"Python interpreter"** dropdown
3. Επίλεξε ή πρόσθεσε interpreter από `.venv\Scripts\python.exe`

### Verify

Μετά την ρύθμιση, στο PyCharm θα πρέπει να βλέπεις:
- Στο Python interpreter selector (κάτω δεξιά): `.venv` ή το path προς `.venv\Scripts\python.exe`

### Alternative: Run from Terminal in PyCharm

Αν δεν θες να αλλάξεις interpreter, μπορείς να τρέξεις από το PyCharm Terminal:

1. **View → Tool Windows → Terminal** (ή `Alt+F12`)
2. Γράφεις:
   ```bash
   .venv\Scripts\activate
   python data_collector.py
   ```


