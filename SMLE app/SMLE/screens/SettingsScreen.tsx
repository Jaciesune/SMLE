import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Alert } from 'react-native';
import { Picker } from '@react-native-picker/picker';
import AsyncStorage from '@react-native-async-storage/async-storage';

type Props = {
  setDefaultModel: (model: string | null) => void;
};

const SettingsScreen: React.FC<Props> = ({ setDefaultModel }) => {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  // Lista dostępnych modeli (taka sama jak w HomeScreen) z opcją "Brak"
  const models = [
    'Brak',
    'DrugyTestMCNN_20250404_173029_checkpoint.pth',
    'ModelYOLO_20250405_123456.pth',
    'FasterRCNN_20250406_789012.pth',
    'YOLOv8_20250407_654321.pth',
  ];

  // Ładowanie domyślnego modelu z AsyncStorage przy starcie
  useEffect(() => {
    const loadDefaultModel = async () => {
      try {
        const storedModel = await AsyncStorage.getItem('defaultModel');
        if (storedModel) {
          setSelectedModel(storedModel);
          setDefaultModel(storedModel);
        } else {
          setSelectedModel('Brak');
          setDefaultModel(null);
        }
      } catch (error) {
        console.log('Błąd podczas ładowania domyślnego modelu:', error);
      }
    };
    loadDefaultModel();
  }, [setDefaultModel]);

  // Zapisywanie domyślnego modelu
  const saveDefaultModel = async (model: string) => {
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

      {/* Wybór domyślnego modelu */}
      <Text style={styles.sectionTitle}>Domyślny model:</Text>
      <Picker
        selectedValue={selectedModel}
        onValueChange={(itemValue) => saveDefaultModel(itemValue)}
        style={styles.picker}
      >
        {models.map((model) => (
          <Picker.Item key={model} label={model} value={model} />
        ))}
      </Picker>

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
  picker: {
    backgroundColor: '#333',
    color: '#FFF',
    borderRadius: 8,
    marginBottom: 20,
  },
  clearButton: {
    backgroundColor: '#FF4444',
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
});

export default SettingsScreen;