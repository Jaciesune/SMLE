import React, { useState, useEffect } from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity, PermissionsAndroid, Platform, FlatList } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';

type Props = {
  selectedModel: string | null;
};

type Model = {
  name: string;
  algorithm: string;
};

const HomeScreen: React.FC<Props> = ({ selectedModel: initialModel }) => {
  const [photo, setPhoto] = useState<string | null>(null);
  const [photoHistory, setPhotoHistory] = useState<string[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('MCNN');
  const [selectedModel, setSelectedModel] = useState<string>(
    initialModel || 'DrugyTestMCNN_20250404_173029_checkpoint.pth'
  );

  const algorithms = ['MCNN', 'YOLO', 'Faster R-CNN'];

  // Lista modeli z przypisanymi algorytmami
  const models: Model[] = [
    { name: 'DrugyTestMCNN_20250404_173029_checkpoint.pth', algorithm: 'MCNN' },
    { name: 'ModelYOLO_20250405_123456.pth', algorithm: 'YOLO' },
    { name: 'FasterRCNN_20250406_789012.pth', algorithm: 'Faster R-CNN' },
    { name: 'YOLOv8_20250407_654321.pth', algorithm: 'YOLO' },
  ];

  // Filtrowanie modeli na podstawie wybranego algorytmu
  const filteredModels = models.filter((model) => model.algorithm === selectedAlgorithm);

  // Ustawianie modelu, jeśli initialModel nie jest null
  useEffect(() => {
    if (initialModel) {
      setSelectedModel(initialModel);
    }
  }, [initialModel]);

  // Automatyczne ustawianie pierwszego modelu po zmianie algorytmu
  useEffect(() => {
    if (filteredModels.length > 0) {
      // Jeśli wybrany model nie pasuje do nowego algorytmu, ustawiamy pierwszy pasujący
      const currentModelMatchesAlgorithm = filteredModels.some(
        (model) => model.name === selectedModel
      );
      if (!currentModelMatchesAlgorithm) {
        setSelectedModel(filteredModels[0].name);
      }
    }
  }, [selectedAlgorithm]);

  // Funkcja do ładowania historii zdjęć
  const loadPhotoHistory = async () => {
    try {
      const storedPhotos = await AsyncStorage.getItem('photoHistory');
      if (storedPhotos) {
        setPhotoHistory(JSON.parse(storedPhotos));
      } else {
        setPhotoHistory([]);
      }
    } catch (error) {
      console.log('Błąd podczas ładowania historii zdjęć:', error);
    }
  };

  // Ładowanie historii zdjęć przy starcie aplikacji
  useEffect(() => {
    loadPhotoHistory();
  }, []);

  // Odświeżanie historii zdjęć, gdy ekran staje się aktywny
  useFocusEffect(
    React.useCallback(() => {
      loadPhotoHistory();
    }, [])
  );

  const savePhotoToHistory = async (uri: string) => {
    try {
      const updatedHistory = [uri, ...photoHistory].slice(0, 10);
      await AsyncStorage.setItem('photoHistory', JSON.stringify(updatedHistory));
      setPhotoHistory(updatedHistory);
    } catch (error) {
      console.log('Błąd podczas zapisywania zdjęcia:', error);
    }
  };

  const requestCameraPermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.CAMERA,
          {
            title: 'Prośba o dostęp do aparatu',
            message: 'Aplikacja potrzebuje dostępu do aparatu, aby zrobić zdjęcie.',
            buttonNeutral: 'Zapytaj później',
            buttonNegative: 'Anuluj',
            buttonPositive: 'OK',
          }
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        console.warn(err);
        return false;
      }
    }
    return true;
  };

  const requestStoragePermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const granted = await PermissionsAndroid.request(
          PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE,
          {
            title: 'Prośba o dostęp do pamięci',
            message: 'Aplikacja potrzebuje dostępu do pamięci, aby zapisać zdjęcie.',
            buttonNeutral: 'Zapytaj później',
            buttonNegative: 'Anuluj',
            buttonPositive: 'OK',
          }
        );
        return granted === PermissionsAndroid.RESULTS.GRANTED;
      } catch (err) {
        console.warn(err);
        return false;
      }
    }
    return true;
  };

  const takePhoto = async () => {
    const cameraGranted = await requestCameraPermission();
    const storageGranted = await requestStoragePermission();

    if (cameraGranted && storageGranted) {
      launchCamera({ mediaType: 'photo', saveToPhotos: true }, (response) => {
        if (response.didCancel) {
          console.log('Anulowano');
        } else if (response.errorCode) {
          console.log('Błąd: ', response.errorMessage);
        } else if (response.assets) {
          const uri = response.assets[0].uri;
          setPhoto(uri);
          savePhotoToHistory(uri);
        }
      });
    } else {
      console.log('Brak uprawnień do aparatu lub pamięci');
    }
  };

  const pickPhoto = async () => {
    const storageGranted = await requestStoragePermission();

    if (storageGranted) {
      launchImageLibrary({ mediaType: 'photo' }, (response) => {
        if (response.didCancel) {
          console.log('Anulowano');
        } else if (response.errorCode) {
          console.log('Błąd: ', response.errorMessage);
        } else if (response.assets) {
          const uri = response.assets[0].uri;
          setPhoto(uri);
          savePhotoToHistory(uri);
        }
      });
    } else {
      console.log('Brak uprawnień do pamięci');
    }
  };

  const changePhoto = () => {
    setPhoto(null);
  };

  const startAnalysis = () => {
    console.log('Rozpoczynam analizę...');
    console.log('Algorytm:', selectedAlgorithm);
    console.log('Model:', selectedModel);
    console.log('Zdjęcie:', photo);
  };

  const renderPhotoItem = ({ item }: { item: string }) => (
    <TouchableOpacity onPress={() => setPhoto(item)}>
      <Image source={{ uri: item }} style={styles.historyImage} />
    </TouchableOpacity>
  );

  return (
    <View style={styles.container}>
      <View style={styles.leftContainer}>
        {photo ? (
          <>
            <Image source={{ uri: photo }} style={styles.image} />
            <TouchableOpacity style={styles.changeButton} onPress={changePhoto}>
              <Text style={styles.buttonText}>Zmień zdjęcie</Text>
            </TouchableOpacity>
          </>
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>Brak zdjęcia</Text>
            <TouchableOpacity style={styles.button} onPress={takePhoto}>
              <Text style={styles.buttonText}>Zrób zdjęcie</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={pickPhoto}>
              <Text style={styles.buttonText}>Wybierz z galerii</Text>
            </TouchableOpacity>
          </View>
        )}
        {photoHistory.length > 0 && (
          <View style={styles.historyContainer}>
            <Text style={styles.historyTitle}>Historia zdjęć:</Text>
            <FlatList
              horizontal
              data={photoHistory}
              renderItem={renderPhotoItem}
              keyExtractor={(item, index) => index.toString()}
              showsHorizontalScrollIndicator={false}
              style={styles.historyList}
            />
          </View>
        )}
      </View>

      <View style={styles.rightContainer}>
        <Text style={styles.sectionTitle}>Wybierz algorytm:</Text>
        <Picker
          selectedValue={selectedAlgorithm}
          onValueChange={(itemValue) => setSelectedAlgorithm(itemValue)}
          style={styles.picker}
        >
          {algorithms.map((algo) => (
            <Picker.Item key={algo} label={algo} value={algo} />
          ))}
        </Picker>

        <Text style={styles.sectionTitle}>Wybierz model:</Text>
        <Picker
          selectedValue={selectedModel}
          onValueChange={(itemValue) => setSelectedModel(itemValue)}
          style={styles.picker}
        >
          {filteredModels.length > 0 ? (
            filteredModels.map((model) => (
              <Picker.Item key={model.name} label={model.name} value={model.name} />
            ))
          ) : (
            <Picker.Item label="Brak dostępnych modeli" value="" />
          )}
        </Picker>

        <TouchableOpacity
          style={[styles.analyzeButton, { opacity: filteredModels.length > 0 ? 1 : 0.5 }]}
          onPress={startAnalysis}
          disabled={filteredModels.length === 0}
        >
          <Text style={styles.buttonText}>Rozpocznij analizę</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: '#121212',
    padding: 10,
  },
  leftContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingRight: 10,
  },
  rightContainer: {
    flex: 1,
    justifyContent: 'center',
    paddingLeft: 10,
  },
  image: {
    width: 150,
    height: 150,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#00A1D6',
  },
  placeholder: {
    width: 150,
    height: 150,
    backgroundColor: '#333',
    borderRadius: 10,
    justifyContent: 'center',
    alignItems: 'center',
  },
  placeholderText: {
    color: '#888',
    marginBottom: 10,
  },
  button: {
    backgroundColor: '#00A1D6',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 8,
    marginVertical: 5,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  changeButton: {
    backgroundColor: '#00A1D6',
    paddingVertical: 8,
    paddingHorizontal: 15,
    borderRadius: 8,
    marginTop: 10,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  analyzeButton: {
    backgroundColor: '#00A1D6',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 20,
    alignItems: 'center',
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
  sectionTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  picker: {
    backgroundColor: '#333',
    color: '#FFF',
    borderRadius: 8,
    marginBottom: 20,
  },
  historyContainer: {
    marginTop: 20,
    alignItems: 'center',
  },
  historyTitle: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  historyList: {
    maxHeight: 100,
  },
  historyImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
    marginHorizontal: 5,
    borderWidth: 1,
    borderColor: '#00A1D6',
  },
});

export default HomeScreen;