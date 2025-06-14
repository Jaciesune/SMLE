import React, { useState, useEffect } from 'react';
import { View, Text, Image, StyleSheet, TouchableOpacity, PermissionsAndroid, Platform, FlatList, Modal, Alert, ActivityIndicator } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import { launchCamera, launchImageLibrary, ImagePickerResponse, Asset } from 'react-native-image-picker';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect, useNavigation } from '@react-navigation/native';
import { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import PushNotification from 'react-native-push-notification';
import ReactNativeBlobUtil from 'react-native-blob-util';
import { encode } from 'base-64';
import { BottomTabParamList } from '../types/navigation';
import { getApiUrl } from '../utils/api';

// Typy dla nawigacji
type NavigationProp = BottomTabNavigationProp<BottomTabParamList, 'Home'>;

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
  const navigation = useNavigation<NavigationProp>();
  const [photos, setPhotos] = useState<string[]>([]);
  const [photoHistory, setPhotoHistory] = useState<string[]>([]);
  const [algorithms, setAlgorithms] = useState<string[]>([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<string | null>(null);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [modalVisible, setModalVisible] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<{
    pipesDetected: number[];
    totalPipes: number;
    photosAfter: string[];
  } | null>(null);
  const [apiUrl, setApiUrl] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);

  // Wczytaj API_URL i przekieruj na Settings, jeśli nie ma zapisanego IP
  useEffect(() => {
    let isMounted = true;

    const initializeApiUrl = async () => {
      try {
        setLoading(true);
        const url = await getApiUrl();
        if (isMounted) {
          setApiUrl(url);

          const storedIp = await AsyncStorage.getItem('serverIp');
          if (!storedIp && isMounted) {
            navigation.navigate('Settings');
          }
        }
      } catch (error) {
        console.log('Błąd podczas inicjalizacji API_URL:', error);
        if (isMounted) {
          Alert.alert('Błąd', 'Nie udało się zainicjalizować adresu API.');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    initializeApiUrl();

    return () => {
      isMounted = false;
    };
  }, [navigation]);

  // Pobieranie listy algorytmów
  useEffect(() => {
    let isMounted = true;

    const fetchAlgorithms = async () => {
      if (!apiUrl) return;

      try {
        setLoading(true);
        console.log('Pobieranie algorytmów z:', `${apiUrl}/detect_algorithms`);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);
        const response = await fetch(`${apiUrl}/detect_algorithms`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`Błąd HTTP: ${response.status} - ${response.statusText}`);
        }

        const data: string[] = await response.json();
        console.log('Dane algorytmów:', data);
        if (isMounted) {
          setAlgorithms(data);
          if (data.length > 0) {
            setSelectedAlgorithm(data[0]);
          } else {
            setSelectedAlgorithm(null);
            Alert.alert('Błąd', 'Backend nie zwrócił żadnych algorytmów. Sprawdź konfigurację serwera.');
          }
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
        console.log('Błąd podczas pobierania algorytmów:', errorMessage);
        if (isMounted) {
          Alert.alert('Błąd', `Nie udało się pobrać listy algorytmów: ${errorMessage}`);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchAlgorithms();

    return () => {
      isMounted = false;
    };
  }, [apiUrl]);

  // Pobieranie listy modeli dla wybranego algorytmu
  useEffect(() => {
    let isMounted = true;

    const fetchModels = async () => {
      if (!selectedAlgorithm || !apiUrl) return;

      try {
        setLoading(true);
        console.log('Pobieranie modeli z:', `${apiUrl}/detect_model_versions/${selectedAlgorithm}`);
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000);
        const response = await fetch(`${apiUrl}/detect_model_versions/${selectedAlgorithm}`, {
          signal: controller.signal,
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`Błąd HTTP: ${response.status}`);
        }

        const data: string[] = await response.json();
        if (isMounted) {
          setAvailableModels(data);
          if (data.length > 0) {
            setSelectedModel(data[0]);
          } else {
            setSelectedModel(null);
            Alert.alert('Błąd', `Nie znaleziono modeli dla algorytmu ${selectedAlgorithm}.`);
          }
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Nieznany błąd';
        console.log('Błąd podczas pobierania modeli:', errorMessage);
        if (isMounted) {
          Alert.alert('Błąd', `Nie udało się pobrać listy modeli dla ${selectedAlgorithm}: ${errorMessage}`);
          setSelectedModel(null);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    fetchModels();

    return () => {
      isMounted = false;
    };
  }, [selectedAlgorithm, apiUrl]);

  const loadPhotoHistory = async () => {
    try {
      setLoading(true);
      const storedPhotos = await AsyncStorage.getItem('photoHistory');
      const parsedPhotos = storedPhotos ? JSON.parse(storedPhotos) as string[] : [];
      setPhotoHistory(parsedPhotos);
    } catch (error) {
      console.log('Błąd podczas ładowania historii zdjęć:', error);
      Alert.alert('Błąd', 'Nie udało się wczytać historii zdjęć.');
      setPhotoHistory([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let isMounted = true;

    if (isMounted) {
      loadPhotoHistory();
    }

    return () => {
      isMounted = false;
    };
  }, []);

  useFocusEffect(
    React.useCallback(() => {
      let isActive = true;

      const fetchHistory = async () => {
        try {
          setLoading(true);
          const storedPhotos = await AsyncStorage.getItem('photoHistory');
          const parsedPhotos = storedPhotos ? JSON.parse(storedPhotos) as string[] : [];
          if (isActive) {
            setPhotoHistory(parsedPhotos);
          }
        } catch (error) {
          console.log('Błąd podczas ładowania historii zdjęć (useFocusEffect):', error);
          if (isActive) {
            Alert.alert('Błąd', 'Nie udało się wczytać historii zdjęć.');
            setPhotoHistory([]);
          }
        } finally {
          if (isActive) {
            setLoading(false);
          }
        }
      };

      fetchHistory();

      return () => {
        isActive = false;
      };
    }, [])
  );

  const savePhotoToHistory = async (uri: string) => {
    if (!uri) return;

    try {
      setLoading(true);
      const updatedHistory = [...new Set([uri, ...photoHistory])].slice(0, 10);
      await AsyncStorage.setItem('photoHistory', JSON.stringify(updatedHistory));
      setPhotoHistory(updatedHistory);
      console.log('Zapisano zdjęcie do historii:', uri);
    } catch (error) {
      console.log('Błąd podczas zapisywania zdjęcia:', error);
      Alert.alert('Błąd', 'Nie udało się zapisać zdjęcia do historii.');
    } finally {
      setLoading(false);
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
      setLoading(true);
      const storedHistory = await AsyncStorage.getItem('analysisHistory');
      console.log('Odczytano historię z AsyncStorage:', storedHistory);
  
      const history = storedHistory ? JSON.parse(storedHistory) as AnalysisResult[] : [];
      console.log('Sparowana historia:', history);
  
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
      console.log('Nowy wpis do zapisania:', newEntry);
  
      const updatedHistory = [newEntry, ...history].slice(0, 50);
      console.log('Zaktualizowana historia przed zapisem:', updatedHistory);
  
      const serializedHistory = JSON.stringify(updatedHistory);
      console.log('Rozmiar zserializowanej historii (bajty):', new TextEncoder().encode(serializedHistory).length);
  
      await AsyncStorage.setItem('analysisHistory', serializedHistory);
      console.log('Zapisano analizę do historii:', newEntry);
    } catch (error) {
      console.log('Błąd podczas zapisywania historii analizy:', error);
      Alert.alert('Błąd', 'Nie udało się zapisać historii analizy.');
    } finally {
      setLoading(false);
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
        console.warn('Błąd podczas żądania uprawnień do aparatu:', err);
        Alert.alert('Błąd', 'Nie udało się uzyskać dostępu do aparatu.');
        return false;
      }
    }
    return true;
  };

  const requestStoragePermission = async () => {
    if (Platform.OS === 'android') {
      try {
        const permissions: (typeof PermissionsAndroid.PERMISSIONS)[keyof typeof PermissionsAndroid.PERMISSIONS][] = [];
        if (Platform.Version >= 33) {
          permissions.push(PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES);
        } else {
          permissions.push(PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE);
          permissions.push(PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE);
        }
        const granted = await PermissionsAndroid.requestMultiple(permissions);
        if (Platform.Version >= 33) {
          return granted[PermissionsAndroid.PERMISSIONS.READ_MEDIA_IMAGES] === PermissionsAndroid.RESULTS.GRANTED;
        } else {
          return (
            granted[PermissionsAndroid.PERMISSIONS.WRITE_EXTERNAL_STORAGE] === PermissionsAndroid.RESULTS.GRANTED &&
            granted[PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE] === PermissionsAndroid.RESULTS.GRANTED
          );
        }
      } catch (err) {
        console.warn('Błąd podczas żądania uprawnień do pamięci:', err);
        Alert.alert('Błąd', 'Nie udało się uzyskać dostępu do pamięci.');
        return false;
      }
    }
    return true;
  };

  const takePhoto = async () => {
    const cameraGranted = await requestCameraPermission();
    const storageGranted = await requestStoragePermission();
  
    if (cameraGranted && storageGranted) {
      launchCamera({ mediaType: 'photo', saveToPhotos: true }, async (response: ImagePickerResponse) => {
        if (response.didCancel) {
          console.log('Anulowano robienie zdjęcia');
        } else if (response.errorCode) {
          console.log('Błąd podczas robienia zdjęcia:', response.errorMessage);
          Alert.alert('Błąd', `Nie udało się zrobić zdjęcia: ${response.errorMessage}`);
        } else if (response.assets) {
          const asset = response.assets[0];
          if (asset.uri) {
            let persistentUri = asset.uri;
            if (Platform.OS === 'android') {
              try {
                const fileName = `original_${Date.now()}.jpg`;
                const mediaStoreResponse = await ReactNativeBlobUtil.MediaCollection.copyToMediaStore(
                  {
                    name: fileName,
                    parentFolder: 'SMLE/Originals',
                    mimeType: 'image/jpeg',
                  },
                  'Image',
                  asset.uri
                );
                persistentUri = mediaStoreResponse;
                console.log('Zdjęcie zapisane w MediaStore:', persistentUri);
              } catch (error) {
                console.log('Błąd zapisu zdjęcia do MediaStore:', error);
                Alert.alert('Błąd', 'Nie udało się zapisać zdjęcia w galerii.');
                return;
              }
            }
            setPhotos((prev) => [...prev, persistentUri]);
            savePhotoToHistory(persistentUri);
          }
        }
      });
    } else {
      Alert.alert('Błąd', 'Brak uprawnień do aparatu lub pamięci.');
    }
  };
  
  const pickPhotos = async () => {
    const storageGranted = await requestStoragePermission();
  
    if (storageGranted) {
      launchImageLibrary({ mediaType: 'photo', selectionLimit: 0 }, async (response: ImagePickerResponse) => {
        if (response.didCancel) {
          console.log('Anulowano wybieranie zdjęć');
        } else if (response.errorCode) {
          console.log('Błąd podczas wybierania zdjęć:', response.errorMessage);
          Alert.alert('Błąd', `Nie udało się wybrać zdjęć: ${response.errorMessage}`);
        } else if (response.assets) {
          const uris: string[] = [];
          for (const asset of response.assets) {
            if (asset.uri) {
              let persistentUri = asset.uri;
              if (Platform.OS === 'android') {
                try {
                  const fileName = `original_${Date.now()}_${uris.length}.jpg`;
                  const mediaStoreResponse = await ReactNativeBlobUtil.MediaCollection.copyToMediaStore(
                    {
                      name: fileName,
                      parentFolder: 'SMLE/Originals',
                      mimeType: 'image/jpeg',
                    },
                    'Image',
                    asset.uri
                  );
                  persistentUri = mediaStoreResponse;
                  console.log('Zdjęcie zapisane w MediaStore:', persistentUri);
                } catch (error) {
                  console.log('Błąd zapisu zdjęcia do MediaStore:', error);
                  Alert.alert('Błąd', 'Nie udało się zapisać zdjęcia w galerii.');
                  continue;
                }
              }
              uris.push(persistentUri);
            }
          }
          setPhotos((prev) => [...prev, ...uris]);
          uris.forEach((uri) => savePhotoToHistory(uri));
        }
      });
    } else {
      Alert.alert('Błąd', 'Brak uprawnień do pamięci.');
    }
  };

  const removePhoto = (index: number) => {
    setPhotos((prev) => prev.filter((_, i) => i !== index));
  };

  const startAnalysis = async () => {
    if (!apiUrl) {
      Alert.alert('Błąd', 'Proszę najpierw skonfigurować adres IP serwera w ustawieniach.');
      navigation.navigate('Settings');
      return;
    }
  
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
  
    const storageGranted = await requestStoragePermission();
    if (!storageGranted) {
      Alert.alert('Błąd', 'Brak uprawnień do zapisu na urządzeniu.');
      return;
    }
  
    try {
      setLoading(true);
      const analysisPromises = photos.map(async (photo, index) => {
        const formData = new FormData();
        formData.append('image', {
          uri: photo,
          type: 'image/jpeg',
          name: `photo_${index}.jpg`,
        } as any);
        formData.append('algorithm', selectedAlgorithm);
        formData.append('model_version', selectedModel);
  
        console.log('Wysyłanie żądania detekcji dla zdjęcia:', photo);
  
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 15000);
        const response = await fetch(`${apiUrl}/detect_image`, {
          method: 'POST',
          body: formData,
          signal: controller.signal,
        });
        clearTimeout(timeoutId);
  
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Błąd podczas analizy zdjęcia: ${response.status}`);
        }
  
        const detectionsCount = parseInt(response.headers.get('X-Detections-Count') || '0', 10);
  
        const tempFileName = `temp_${Date.now()}_${index}.jpg`;
        const tempFilePath = `${ReactNativeBlobUtil.fs.dirs.CacheDir}/${tempFileName}`;
  
        const arrayBuffer = await response.arrayBuffer();
        const uint8Array = new Uint8Array(arrayBuffer);
        const binaryString = uint8Array.reduce((acc, byte) => acc + String.fromCharCode(byte), '');
        const base64Image = encode(binaryString);
  
        await ReactNativeBlobUtil.fs.writeFile(tempFilePath, base64Image, 'base64');
        console.log('Zdjęcie tymczasowe zapisane w:', tempFilePath);
  
        const fileName = `${detectionsCount}_rur_${Date.now()}_${index}.jpg`;
        let finalUri;
  
        if (Platform.OS === 'android') {
          const mediaStoreResponse = await ReactNativeBlobUtil.MediaCollection.copyToMediaStore(
            {
              name: fileName,
              parentFolder: 'SMLE/Processed',
              mimeType: 'image/jpeg',
            },
            'Image',
            tempFilePath
          );
          console.log('Zdjęcie zapisane w galerii przez Media Store:', mediaStoreResponse);
          finalUri = mediaStoreResponse;
        } else {
          finalUri = `${ReactNativeBlobUtil.fs.dirs.DocumentDir}/${fileName}`;
          await ReactNativeBlobUtil.fs.writeFile(finalUri, base64Image, 'base64');
        }
  
        await ReactNativeBlobUtil.fs.unlink(tempFilePath).catch((err) => {
          console.log('Błąd podczas usuwania pliku tymczasowego:', err);
        });
  
        return {
          detectionsCount,
          finalUri,
        };
      });
  
      const results = await Promise.all(analysisPromises);
  
      const pipesDetected = results.map((result) => result.detectionsCount);
      const photosAfter = results.map((result) => result.finalUri);
      const totalPipes = pipesDetected.reduce((sum, count) => sum + count, 0);
  
      console.log('Photos Before (przed zapisem do historii):', photos);
      console.log('Photos After (przed zapisem do historii):', photosAfter);
  
      setAnalysisResult({
        pipesDetected,
        totalPipes,
        photosAfter,
      });
  
      await saveAnalysisToHistory({
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
        message: `Wykryto ${totalPipes} elementów.`,
        smallIcon: 'ic_notification',
        color: '#00A1D6',
        vibrate: true,
        vibration: 300,
      });
  
      setModalVisible(true);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.log('Błąd podczas analizy:', errorMessage);
      Alert.alert('Błąd', `Nie udało się przeprowadzić analizy: ${errorMessage}`);
    } finally {
      setLoading(false);
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
    <View style={styles.resultPhotoContainer}>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Przed</Text>
        <Image source={{ uri: item.before }} style={styles.resultImage} />
      </View>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Po</Text>
        <Image source={{ uri: item.after }} style={styles.resultImage} />
      </View>
      <Text style={styles.statsText}>Zdjęcie {index + 1}: {item.pipes} elementów</Text>
    </View>
  );

  return (
    <View style={styles.container}>
      {loading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#00A1D6" />
          <Text style={styles.loadingText}>Ładowanie...</Text>
        </View>
      )}
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
                <Text style={styles.modalText}>Algorytm: {selectedAlgorithm ?? 'Nie wybrano'}</Text>
                <Text style={styles.modalText}>Model: {selectedModel ?? 'Nie wybrano'}</Text>
                <FlatList
                  data={analysisResult.photosAfter.map((after, idx) => ({
                    before: photos[idx] ?? '',
                    after,
                    pipes: analysisResult.pipesDetected[idx] ?? 0,
                  }))}
                  renderItem={renderResultPhotoPair}
                  keyExtractor={(item, index) => `result-${index}`}
                />
                <Text style={styles.modalText}>Łącznie: {analysisResult.totalPipes} elementów</Text>
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
            keyExtractor={(item, index) => `photo-${index}`}
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
              keyExtractor={(item, index) => `history-${index}`}
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
            <Picker.Item label="Brak dostępnych algorytmów" value={null} />
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
            <Picker.Item label="Brak dostępnych modeli" value={null} />
          )}
        </Picker>

        <TouchableOpacity
          style={[styles.analyzeButton, { opacity: availableModels.length > 0 && selectedModel ? 1 : 0.5 }]}
          onPress={startAnalysis}
          disabled={!availableModels.length || !selectedModel || loading}
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
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    zIndex: 1000,
  },
  loadingText: {
    color: '#FFF',
    fontSize: 16,
    marginTop: 10,
  },
});

export default HomeScreen;