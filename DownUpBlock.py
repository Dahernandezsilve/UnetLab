from tensorflow.keras.layers import (Concatenate, Conv2D, Dropout,
                                     MaxPooling2D, UpSampling2D)


# ------------------------------------------------------------------
# Bloque de bajada (DownBlock)
# ------------------------------------------------------------------
def DownBlock(X, filters, drop_p: float = 0.0, module_name: str = ''):
    """
    Dos convoluciones 3×3 → MaxPooling 2×2 → Dropout
    """
    X = Conv2D(filters,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               name=f'{module_name}_Conv1')(X)

    X = Conv2D(filters,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               name=f'{module_name}_Conv2')(X)

    X = MaxPooling2D(pool_size=(2, 2),
                     name=f'{module_name}_MaxPool')(X)

    X = Dropout(drop_p,
                name=f'{module_name}_Dropout')(X)
    return X


# ------------------------------------------------------------------
# Bloque de subida (UpBlock)
# ------------------------------------------------------------------
def UpBlock(X, Y, filters, drop_p: float = 0.0, module_name: str = ''):
    """
    Upsampling bilineal → concatenación con skip-connection →
    dos convoluciones 3×3 → Dropout
    """
    X = UpSampling2D(size=(2, 2),
                     interpolation='bilinear',
                     name=f'{module_name}_UpSample')(X)

    X = Concatenate(name=f'{module_name}_Concat')([X, Y])

    X = Conv2D(filters,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               name=f'{module_name}_Conv1')(X)

    X = Conv2D(filters,
               kernel_size=(3, 3),
               padding='same',
               activation='relu',
               name=f'{module_name}_Conv2')(X)

    X = Dropout(drop_p,
                name=f'{module_name}_Dropout')(X)
    return X
