import React, { useState, useEffect } from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity, PermissionsAndroid, Platform, FlatList, Modal } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import PushNotification from 'react-native-push-notification';

type Props = {
  selectedModel: string | null;
};

type Model = {
  name: string;
  algorithm: string;
};

type AnalysisResult = {
  id: string;
  photosBefore: string[]; // Lista zdjęć przed analizą
  photosAfter: string[]; // Lista zdjęć po analizie (na razie symulacja)
  pipesDetected: number[]; // Lista wyników dla każdego zdjęcia
  totalPipes: number; // Suma rur ze wszystkich zdjęć
  confidence: number; // Średnia pewność
  algorithm: string;
  model: string;
  timestamp: string;
};

// Konfiguracja powiadomień
PushNotification.configure({
  onRegister: function (token) {
    console.log('TOKEN:', token);
  },
  onNotification: function (notification) {
    console.log('NOTIFICATION:', notification);
  },
  permissions: {
    alert: true,
    badge: true,
    sound: true,
  },
  popInitialNotification: true,
  requestPermissions: Platform.OS === 'ios',
});

PushNotification.createChannel(
  {
    channelId: 'smle-analysis-channel',
    channelName: 'SMLE Analysis Notifications',
    channelDescription: 'Powiadomienia o zakończeniu analizy w aplikacji SMLE',
    soundName: 'default',
    importance: 4,
    vibrate: true,
  },
  (created) => console.log(`Kanał powiadomień utworzony: ${created}`)
);

const HomeScreen: React.FC<Props> = ({ selectedModel: initialModel }) => {
  const [photos, setPhotos] = useState<string[]>([]); // Lista wybranych zdjęć
  const [photoHistory, setPhotoHistory] = useState<string[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('MCNN');
  const [selectedModel, setSelectedModel] = useState<string>(
    initialModel || 'DrugyTestMCNN_20250404_173029_checkpoint.pth'
  );
  const [modalVisible, setModalVisible] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<{
    pipesDetected: number[];
    totalPipes: number;
    confidence: number;
  } | null>(null);

  const algorithms = ['MCNN', 'YOLO', 'Faster R-CNN'];

  const models: Model[] = [
    { name: 'DrugyTestMCNN_20250404_173029_checkpoint.pth', algorithm: 'MCNN' },
    { name: 'ModelYOLO_20250405_123456.pth', algorithm: 'YOLO' },
    { name: 'FasterRCNN_20250406_789012.pth', algorithm: 'Faster R-CNN' },
    { name: 'YOLOv8_20250407_654321.pth', algorithm: 'YOLO' },
  ];

  const filteredModels = models.filter((model) => model.algorithm === selectedAlgorithm);

  useEffect(() => {
    if (initialModel) {
      setSelectedModel(initialModel);
    }
  }, [initialModel]);

  useEffect(() => {
    if (filteredModels.length > 0) {
      const currentModelMatchesAlgorithm = filteredModels.some(
        (model) => model.name === selectedModel
      );
      if (!currentModelMatchesAlgorithm) {
        setSelectedModel(filteredModels[0].name);
      }
    }
  }, [selectedAlgorithm]);

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

  useEffect(() => {
    loadPhotoHistory();
  }, []);

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
      console.log('Zapisano zdjęcie do historii:', uri);
    } catch (error) {
      console.log('Błąd podczas zapisywania zdjęcia:', error);
    }
  };

  const saveAnalysisToHistory = async (result: {
    photosBefore: string[];
    pipesDetected: number[];
    totalPipes: number;
    confidence: number;
    algorithm: string;
    model: string;
  }) => {
    try {
      const storedHistory = await AsyncStorage.getItem('analysisHistory');
      const history = storedHistory ? JSON.parse(storedHistory) : [];
      const newEntry: AnalysisResult = {
        id: Date.now().toString(),
        photosBefore: result.photosBefore,
        photosAfter: result.photosBefore, // Na razie to samo zdjęcie (symulacja)
        pipesDetected: result.pipesDetected,
        totalPipes: result.totalPipes,
        confidence: result.confidence,
        algorithm: result.algorithm,
        model: result.model,
        timestamp: new Date().toISOString(),
      };
      const updatedHistory = [newEntry, ...history].slice(0, 50); // Ograniczenie do 50 wpisów
      await AsyncStorage.setItem('analysisHistory', JSON.stringify(updatedHistory));
      console.log('Zapisano analizę do historii:', newEntry);
    } catch (error) {
      console.log('Błąd podczas zapisywania historii analizy:', error);
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
          setPhotos((prev) => [...prev, uri]);
          savePhotoToHistory(uri);
        }
      });
    } else {
      console.log('Brak uprawnień do aparatu lub pamięci');
    }
  };

  const pickPhotos = async () => {
    const storageGranted = await requestStoragePermission();

    if (storageGranted) {
      launchImageLibrary({ mediaType: 'photo', selectionLimit: 0 }, (response) => {
        if (response.didCancel) {
          console.log('Anulowano');
        } else if (response.errorCode) {
          console.log('Błąd: ', response.errorMessage);
        } else if (response.assets) {
          const uris = response.assets.map((asset) => asset.uri);
          setPhotos((prev) => [...prev, ...uris]);
          uris.forEach((uri) => savePhotoToHistory(uri));
        }
      });
    } else {
      console.log('Brak uprawnień do pamięci');
    }
  };

  const removePhoto = (index: number) => {
    setPhotos((prev) => prev.filter((_, i) => i !== index));
  };

  const startAnalysis = () => {
    if (photos.length === 0) {
      alert('Proszę wybrać co najmniej jedno zdjęcie przed rozpoczęciem analizy.');
      return;
    }

    // Symulacja wyników analizy dla każdego zdjęcia
    const pipesDetected = photos.map(() => Math.floor(Math.random() * 10) + 1); // Losowa liczba rur (1-10) dla każdego zdjęcia
    const totalPipes = pipesDetected.reduce((sum, num) => sum + num, 0); // Suma rur
    const confidence = Math.floor(Math.random() * 30) + 70; // Losowy procent pewności (70-99%)

    setAnalysisResult({
      pipesDetected,
      totalPipes,
      confidence,
    });

    // Zapisz wyniki do historii
    saveAnalysisToHistory({
      photosBefore: photos,
      pipesDetected,
      totalPipes,
      confidence,
      algorithm: selectedAlgorithm,
      model: selectedModel,
    });

    // Wyświetlenie powiadomienia
    PushNotification.localNotification({
      channelId: 'smle-analysis-channel',
      title: 'Analiza zakończona',
      message: `Wykryto ${totalPipes} rur z pewnością ${confidence}%`,
      smallIcon: 'ic_notification',
      color: '#00A1D6',
      vibrate: true,
      vibration: 300,
    });

    // Pokaż modal z wynikami
    setModalVisible(true);
  };

  const closeModal = () => {
    setModalVisible(false);
    setAnalysisResult(null);
    setPhotos([]); // Wyczyść wybrane zdjęcia po zamknięciu modala
  };

  const renderPhotoItem = ({ item }: { item: string }) => (
    <TouchableOpacity onPress={() => setPhotos((prev) => [...prev, item])}>
      <Image source={{ uri: item }} style={styles.historyImage} />
    </TouchableOpacity>
  );

  const renderSelectedPhoto = ({ item, index }: { item: string; index: number }) => (
    <View style={styles.selectedPhotoContainer}>
      <Image source={{ uri: item }} style={styles.image} />
      <TouchableOpacity
        style={styles.removeButton}
        onPress={() => removePhoto(index)}
      >
        <Text style={styles.buttonText}>X</Text>
      </TouchableOpacity>
    </View>
  );

  return (
    <View style={styles.container}>
      <Modal
        animationType="slide"
        transparent={true}
        visible={modalVisible}
        onRequestClose={closeModal}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Wyniki analizy</Text>
            {analysisResult && (
              <>
                <Text style={styles.modalText}>Algorytm: {selectedAlgorithm}</Text>
                <Text style={styles.modalText}>Model: {selectedModel}</Text>
                {analysisResult.pipesDetected.map((pipes, index) => (
                  <Text key={index} style={styles.modalText}>
                    Zdjęcie {index + 1}: {pipes} rur
                  </Text>
                ))}
                <Text style={styles.modalText}>Łącznie: {analysisResult.totalPipes} rur</Text>
                <Text style={styles.modalText}>Pewność: {analysisResult.confidence}%</Text>
              </>
            )}
            <TouchableOpacity style={styles.closeButton} onPress={closeModal}>
              <Text style={styles.buttonText}>Zamknij</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      <View style={styles.leftContainer}>
        {photos.length > 0 ? (
          <FlatList
            data={photos}
            renderItem={renderSelectedPhoto}
            keyExtractor={(item, index) => index.toString()}
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.selectedPhotosList}
          />
        ) : (
          <View style={styles.placeholder}>
            <Text style={styles.placeholderText}>Brak wybranych zdjęć</Text>
            <TouchableOpacity style={styles.button} onPress={takePhoto}>
              <Text style={styles.buttonText}>Zrób zdjęcie</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={pickPhotos}>
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
  selectedPhotosList: {
    maxHeight: 180,
  },
  selectedPhotoContainer: {
    position: 'relative',
    marginRight: 10,
  },
  image: {
    width: 150,
    height: 150,
    borderRadius: 10,
    borderWidth: 2,
    borderColor: '#00A1D6',
  },
  removeButton: {
    position: 'absolute',
    top: 5,
    right: 5,
    backgroundColor: '#FF4444',
    borderRadius: 15,
    width: 30,
    height: 30,
    justifyContent: 'center',
    alignItems: 'center',
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
  modalOverlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  modalContent: {
    width: '80%',
    backgroundColor: '#1E1E1E',
    borderRadius: 10,
    padding: 20,
    alignItems: 'center',
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FFD700',
    marginBottom: 20,
  },
  modalText: {
    fontSize: 18,
    color: '#FFF',
    marginBottom: 10,
  },
  closeButton: {
    backgroundColor: '#FF4444',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginTop: 20,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
});

export default HomeScreen;