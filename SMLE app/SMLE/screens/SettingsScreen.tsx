import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert, TextInput } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useNavigation } from '@react-navigation/native';
import { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import { BottomTabParamList } from '../types/navigation';
import { getApiUrl } from '../utils/api';

// Typy dla nawigacji
type NavigationProp = BottomTabNavigationProp<BottomTabParamList, 'Settings'>;

type Props = {
  setDefaultModel: (model: string | null) => void;
};

const SettingsScreen: React.FC<Props> = ({ setDefaultModel }) => {
  const navigation = useNavigation<NavigationProp>();
  const [selectedModel, setSelectedModel] = useState<string>('Brak'); // Ustawiamy 'Brak' jako domyślną wartość, aby uniknąć null
  const [serverIp, setServerIp] = useState<string>('');
  const [apiUrl, setApiUrl] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);

  // Wczytaj API_URL i zapisane IP
  useEffect(() => {
    let isMounted = true;

    const initializeApiUrl = async () => {
      try {
        const url = await getApiUrl();
        if (isMounted) {
          setApiUrl(url);

          const storedIp = await AsyncStorage.getItem('serverIp');
          if (storedIp && isMounted) {
            setServerIp(storedIp);
          }
        }
      } catch (error) {
        console.log('Błąd podczas inicjalizacji API_URL:', error);
      }
    };

    initializeApiUrl();

    return () => {
      isMounted = false;
    };
  }, []);

  // Wczytaj domyślny model z AsyncStorage
  useEffect(() => {
    let isMounted = true;

    const loadDefaultModel = async () => {
      try {
        const storedModel = await AsyncStorage.getItem('defaultModel');
        if (isMounted) {
          if (storedModel) {
            setSelectedModel(storedModel);
            setDefaultModel(storedModel);
          } else {
            setSelectedModel('Brak');
            setDefaultModel(null);
          }
        }
      } catch (error) {
        console.log('Błąd podczas ładowania domyślnego modelu:', error);
        if (isMounted) {
          Alert.alert('Błąd', 'Nie udało się wczytać domyślnego modelu.');
        }
      }
    };

    loadDefaultModel();

    return () => {
      isMounted = false;
    };
  }, [setDefaultModel]);

  // Pobierz listę modeli z serwera
  useEffect(() => {
    let isMounted = true;

    const fetchModels = async () => {
      if (!apiUrl) return;

      try {
        // Tworzymy AbortController dla pierwszego żądania
        const controller1 = new AbortController();
        const timeoutId1 = setTimeout(() => controller1.abort(), 10000); // Timeout 10 sekund
        const algorithmsResponse = await fetch(`${apiUrl}/detect_algorithms`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
          },
          signal: controller1.signal,
        });
        clearTimeout(timeoutId1); // Wyczyść timeout

        if (!algorithmsResponse.ok) {
          throw new Error(`Błąd pobierania algorytmów: ${algorithmsResponse.status}`);
        }

        const algorithms: string[] = await algorithmsResponse.json();
        const allModels: string[] = ['Brak'];

        for (const algorithm of algorithms) {
          try {
            // Tworzymy AbortController dla drugiego żądania
            const controller2 = new AbortController();
            const timeoutId2 = setTimeout(() => controller2.abort(), 10000); // Timeout 10 sekund
            const modelsResponse = await fetch(`${apiUrl}/detect_model_versions/${algorithm}`, {
              method: 'GET',
              signal: controller2.signal,
            });
            clearTimeout(timeoutId2); // Wyczyść timeout

            if (!modelsResponse.ok) {
              throw new Error(`Błąd pobierania modeli dla ${algorithm}: ${modelsResponse.status}`);
            }

            const models: string[] = await modelsResponse.json();
            allModels.push(...models);
          } catch (error) {
            const errorMessage = error instanceof Error ? error.message : String(error);
            console.log(`Błąd pobierania modeli dla ${algorithm}:`, errorMessage);
            if (isMounted) {
              Alert.alert(
                'Ostrzeżenie',
                `Nie udało się pobrać modeli dla algorytmu ${algorithm}: ${errorMessage}`
              );
            }
          }
        }

        if (isMounted) {
          setAvailableModels(allModels);
          // Upewnij się, że selectedModel jest w availableModels
          if (selectedModel && !allModels.includes(selectedModel)) {
            setSelectedModel('Brak');
            setDefaultModel(null);
          }
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.log('Błąd podczas pobierania modeli:', errorMessage);
        if (isMounted) {
          Alert.alert('Błąd', `Nie udało się pobrać listy modeli: ${errorMessage}`);
        }
      }
    };

    fetchModels();

    return () => {
      isMounted = false;
    };
  }, [apiUrl, selectedModel, setDefaultModel]);

  // Zapisywanie IP serwera
  const saveServerIp = async () => {
    try {
      const ipRegex = /^(http:\/\/)?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(:\d{1,5})?$/;
      const cleanedIp = serverIp.startsWith('http://') ? serverIp : `http://${serverIp}`;
      if (!ipRegex.test(cleanedIp)) {
        Alert.alert('Błąd', 'Proszę wpisać poprawny adres IP (np. 192.168.0.3:8000)');
        return;
      }

      await AsyncStorage.setItem('serverIp', cleanedIp);
      setApiUrl(cleanedIp);
      Alert.alert('Sukces', 'Adres IP serwera został zapisany.');
    } catch (error) {
      console.log('Błąd podczas zapisywania IP serwera:', error);
      Alert.alert('Błąd', 'Nie udało się zapisać adresu IP.');
    }
  };

  // Zapisywanie domyślnego modelu
  const saveDefaultModelHandler = async (model: string) => {
    try {
      if (model === 'Brak') {
        await AsyncStorage.removeItem('defaultModel');
        setSelectedModel('Brak');
        setDefaultModel(null);
        Alert.alert('Sukces', 'Domyślny model został usunięty.');
      } else {
        await AsyncStorage.setItem('defaultModel', model);
        setSelectedModel(model);
        setDefaultModel(model);
        Alert.alert('Sukces', `Domyślny model ustawiony na: ${model}`);
      }
    } catch (error) {
      console.log('Błąd podczas zapisywania domyślnego modelu:', error);
      Alert.alert('Błąd', 'Nie udało się zapisać domyślnego modelu.');
    }
  };

  // Czyszczenie historii zdjęć
  const clearPhotoHistory = async () => {
    try {
      await AsyncStorage.removeItem('photoHistory');
      Alert.alert('Sukces', 'Historia zdjęć została wyczyszczona.');
    } catch (error) {
      console.log('Błąd podczas czyszczenia historii zdjęć:', error);
      Alert.alert('Błąd', 'Nie udało się wyczyścić historii zdjęć.');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Ustawienia</Text>

      {/* Ustawianie IP serwera */}
      <Text style={styles.sectionTitle}>Adres IP serwera:</Text>
      <TextInput
        style={styles.input}
        value={serverIp}
        onChangeText={setServerIp}
        placeholder="http://192.168.0.3:8000"
        placeholderTextColor="#888"
        autoCapitalize="none"
        keyboardType="url"
      />
      <TouchableOpacity style={styles.saveButton} onPress={saveServerIp}>
        <Text style={styles.buttonText}>Zapisz IP serwera</Text>
      </TouchableOpacity>

      {/* Przycisk do czyszczenia historii zdjęć */}
      <TouchableOpacity style={styles.clearButton} onPress={clearPhotoHistory}>
        <Text style={styles.buttonText}>Wyczyść historię zdjęć</Text>
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
    padding: 20,
  },
  title: {
    fontSize: 28,
    color: '#FFF',
    fontWeight: 'bold',
    textAlign: 'center',
    marginVertical: 20,
  },
  sectionTitle: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  input: {
    backgroundColor: '#333',
    color: '#FFF',
    borderRadius: 8,
    padding: 10,
    marginBottom: 10,
    fontSize: 16,
  },
  picker: {
    backgroundColor: '#333',
    color: '#FFF',
    borderRadius: 8,
    marginBottom: 20,
  },
  saveButton: {
    backgroundColor: '#00A1D6',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginBottom: 20,
    alignItems: 'center',
  },
  clearButton: {
    backgroundColor: '#FF4444',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginBottom: 20,
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
});

export default SettingsScreen;