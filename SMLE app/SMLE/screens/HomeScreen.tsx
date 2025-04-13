import React, { useState, useEffect } from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity, PermissionsAndroid, Platform, FlatList, Modal, Alert } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { launchCamera, launchImageLibrary } from 'react-native-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import PushNotification from 'react-native-push-notification';
import ReactNativeBlobUtil from 'react-native-blob-util';

// Typy dla wyników analizy
type AnalysisResult = {
  id: string;
  photosBefore: string[];
  photosAfter: string[];
  pipesDetected: number[];
  totalPipes: number;
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

const HomeScreen: React.FC = () => {
  const [photos, setPhotos] = useState<string[]>([]);
  const [photoHistory, setPhotoHistory] = useState<string[]>([]);
  const [algorithms, setAlgorithms] = useState<string[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [modalVisible, setModalVisible] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<{
    pipesDetected: number[];
    totalPipes: number;
    photosAfter: string[];
  } | null>(null);

  // Adres API – dostosuj do swojego środowiska
  const API_URL = 'http://192.168.0.3:8000'; // Ustaw adres ip swojego serwera


  // Pobieranie listy algorytmów
  useEffect(() => {
    const fetchModels = async () => {
      console.log('Wybrany algorytm:', selectedAlgorithm); // Dodaj log
      if (!selectedAlgorithm) return;
      try {
        const response = await fetch(`${API_URL}/detect_model_versions/${selectedAlgorithm}`);
        const data = await response.json();
        setAvailableModels(data);
        if (data.length > 0) {
          setSelectedModel(data[0]);
        } else {
          setSelectedModel('');
        }
      } catch (error) {
        console.log('Błąd podczas pobierania modeli:', error);
        Alert.alert('Błąd', `Nie udało się pobrać listy modeli dla ${selectedAlgorithm}.`);
      }
    };
    fetchModels();
  }, [selectedAlgorithm]);

  useEffect(() => {
    const fetchAlgorithms = async () => {
      try {
        console.log('Pobieranie algorytmów z:', `${API_URL}/detect_algorithms`);
        const response = await fetch(`${API_URL}/detect_algorithms`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
        });
        if (!response.ok) {
          throw new Error(`Błąd HTTP: ${response.status} - ${response.statusText}`);
        }
        const data = await response.json();
        console.log('Dane algorytmów:', data);
        setAlgorithms(data);
        if (data.length > 0) {
          setSelectedAlgorithm(data[0]);
        } else {
          Alert.alert('Błąd', 'Backend nie zwrócił żadnych algorytmów. Sprawdź konfigurację serwera.');
        }
      } catch (error) {
        console.log('Błąd podczas pobierania algorytmów:', error);
        const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
        Alert.alert('Błąd', `Nie udało się pobrać listy algorytmów: ${errorMessage}`);
      }
    };
    fetchAlgorithms();
  }, []);

  // Pobieranie listy modeli dla wybranego algorytmu
  useEffect(() => {
    const fetchModels = async () => {
      if (!selectedAlgorithm) return;
      try {
        const response = await fetch(`${API_URL}/detect_model_versions/${selectedAlgorithm}`);
        const data = await response.json();
        setAvailableModels(data);
        if (data.length > 0) {
          setSelectedModel(data[0]);
        } else {
          setSelectedModel('');
        }
      } catch (error) {
        console.log('Błąd podczas pobierania modeli:', error);
        Alert.alert('Błąd', `Nie udało się pobrać listy modeli dla ${selectedAlgorithm}.`);
      }
    };
    fetchModels();
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
    if (!uri) return;
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
    photosAfter: string[];
    pipesDetected: number[];
    totalPipes: number;
    algorithm: string;
    model: string;
  }) => {
    try {
      const storedHistory = await AsyncStorage.getItem('analysisHistory');
      const history = storedHistory ? JSON.parse(storedHistory) : [];
      const newEntry: AnalysisResult = {
        id: Date.now().toString(),
        photosBefore: result.photosBefore,
        photosAfter: result.photosAfter,
        pipesDetected: result.pipesDetected,
        totalPipes: result.totalPipes,
        algorithm: result.algorithm,
        model: result.model,
        timestamp: new Date().toISOString(),
      };
      const updatedHistory = [newEntry, ...history].slice(0, 50);
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
          const uri = response.assets[0]?.uri;
          if (uri) {
            setPhotos((prev) => [...prev, uri]);
            savePhotoToHistory(uri);
          }
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
          const uris = response.assets
            .map((asset) => asset.uri)
            .filter((uri): uri is string => uri !== undefined);
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

  const startAnalysis = async () => {
    if (photos.length === 0) {
      Alert.alert('Błąd', 'Proszę wybrać co najmniej jedno zdjęcie przed rozpoczęciem analizy.');
      return;
    }
    if (!selectedAlgorithm) {
      Alert.alert('Błąd', 'Proszę wybrać algorytm do analizy.');
      return;
    }
    if (!selectedModel) {
      Alert.alert('Błąd', 'Proszę wybrać model do analizy.');
      return;
    }

    try {
      const pipesDetected: number[] = [];
      const photosAfter: string[] = [];
      let totalPipes = 0;

      for (const photo of photos) {
        const formData = new FormData();
        formData.append('image', {
          uri: photo,
          type: 'image/jpeg',
          name: 'photo.jpg',
        } as any);

        formData.append('algorithm', selectedAlgorithm);
        formData.append('model_version', selectedModel);

        const response = await fetch(`${API_URL}/detect_image`, {
          method: 'POST',
          body: formData,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || 'Błąd podczas analizy zdjęcia');
        }

        const detectionsCount = parseInt(response.headers.get('X-Detections-Count') || '0', 10);
        pipesDetected.push(detectionsCount);
        totalPipes += detectionsCount;

        const fileName = `detected_${Date.now()}_${photos.indexOf(photo)}.jpg`;
        const filePath = `${ReactNativeBlobUtil.fs.dirs.CacheDir}/${fileName}`;

        await ReactNativeBlobUtil.config({
          path: filePath,
        })
          .fetch('GET', response.url)
          .then((res) => {
            console.log('Zdjęcie zapisane w:', res.path());
          });

        photosAfter.push(`file://${filePath}`);
      }

      setAnalysisResult({
        pipesDetected,
        totalPipes,
        photosAfter,
      });

      saveAnalysisToHistory({
        photosBefore: photos,
        photosAfter,
        pipesDetected,
        totalPipes,
        algorithm: selectedAlgorithm,
        model: selectedModel,
      });

      PushNotification.localNotification({
        channelId: 'smle-analysis-channel',
        title: 'Analiza zakończona',
        message: `Wykryto ${totalPipes} rur.`,
        smallIcon: 'ic_notification',
        color: '#00A1D6',
        vibrate: true,
        vibration: 300,
      });

      setModalVisible(true);
    } catch (error: unknown) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log('Błąd podczas analizy:', errorMessage);
      Alert.alert('Błąd', `Nie udało się przeprowadzić analizy: ${errorMessage}`);
    }
  };

  const closeModal = () => {
    setModalVisible(false);
    setAnalysisResult(null);
    setPhotos([]);
  };

  const renderPhotoItem = ({ item }: { item: string }) => (
    <TouchableOpacity onPress={() => setPhotos((prev) => [...prev, item])}>
      <Image source={{ uri: item }} style={styles.historyImage} />
    </TouchableOpacity>
  );

  const renderSelectedPhoto = ({ item, index }: { item: string; index: number }) => (
    <View style={styles.selectedPhotoContainer}>
      <Image source={{ uri: item }} style={styles.image} />
      <TouchableOpacity style={styles.removeButton} onPress={() => removePhoto(index)}>
        <Text style={styles.buttonText}>X</Text>
      </TouchableOpacity>
    </View>
  );

  const renderResultPhotoPair = ({ item, index }: { item: { before: string; after: string; pipes: number }; index: number }) => (
    <View key={index} style={styles.resultPhotoContainer}>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Przed</Text>
        <Image source={{ uri: item.before }} style={styles.resultImage} />
      </View>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Po</Text>
        <Image source={{ uri: item.after }} style={styles.resultImage} />
      </View>
      <Text style={styles.statsText}>Zdjęcie {index + 1}: {item.pipes} rur</Text>
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
                <FlatList
                  data={analysisResult.photosAfter.map((after, idx) => ({
                    before: photos[idx],
                    after,
                    pipes: analysisResult.pipesDetected[idx],
                  }))}
                  renderItem={renderResultPhotoPair}
                  keyExtractor={(item, index) => index.toString()}
                />
                <Text style={styles.modalText}>Łącznie: {analysisResult.totalPipes} rur</Text>
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
          {algorithms.length > 0 ? (
            algorithms.map((algo) => (
              <Picker.Item key={algo} label={algo} value={algo} />
            ))
          ) : (
            <Picker.Item label="Brak dostępnych algorytmów" value="" />
          )}
        </Picker>

        <Text style={styles.sectionTitle}>Wybierz model:</Text>
        <Picker
          selectedValue={selectedModel}
          onValueChange={(itemValue) => setSelectedModel(itemValue)}
          style={styles.picker}
        >
          {availableModels.length > 0 ? (
            availableModels.map((model) => (
              <Picker.Item key={model} label={model} value={model} />
            ))
          ) : (
            <Picker.Item label="Brak dostępnych modeli" value="" />
          )}
        </Picker>

        <TouchableOpacity
          style={[styles.analyzeButton, { opacity: availableModels.length > 0 ? 1 : 0.5 }]}
          onPress={startAnalysis}
          disabled={availableModels.length === 0}
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
    width: '90%',
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
  resultPhotoContainer: {
    marginBottom: 20,
    alignItems: 'center',
  },
  imageWrapper: {
    alignItems: 'center',
    marginBottom: 5,
  },
  imageLabel: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  resultImage: {
    width: 200,
    height: 200,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#00A1D6',
  },
  statsText: {
    color: '#FFF',
    fontSize: 16,
    marginTop: 5,
  },
});

export default HomeScreen;