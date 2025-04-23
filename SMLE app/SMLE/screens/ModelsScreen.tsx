import React, { useState, useEffect } from 'react';
import { View, Text, SectionList, StyleSheet, TouchableOpacity, ActivityIndicator, Alert } from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { BottomTabNavigationProp } from '@react-navigation/bottom-tabs';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { BottomTabParamList } from '../types/navigation';
import { getApiUrl } from '../utils/api';

// Typy dla nawigacji
type NavigationProp = BottomTabNavigationProp<BottomTabParamList, 'Models'>;

// Typy dla danych modelu
type ModelItem = {
  id: string;
  name: string;
  date: string;
  description: string;
};

type SectionData = {
  algorithm: string;
  data: ModelItem[];
};

type Props = {
  setSelectedModel: (model: string) => void;
};

const ModelsScreen: React.FC<Props> = ({ setSelectedModel }) => {
  const navigation = useNavigation<NavigationProp>();
  const [sections, setSections] = useState<SectionData[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [apiUrl, setApiUrl] = useState<string>('');

  // Wczytaj API_URL i przekieruj na Settings, jeśli nie ma zapisanego IP
  useEffect(() => {
    let isMounted = true;

    const initializeApiUrl = async () => {
      try {
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
      }
    };

    initializeApiUrl();

    return () => {
      isMounted = false;
    };
  }, [navigation]);

  // Funkcja do wyodrębnienia daty z nazwy modelu
  const extractDateFromModelName = (modelName: string): string => {
    const regex = /(\d{8})/;
    const match = modelName.match(regex);
    if (match) {
      const dateStr = match[0];
      return `${dateStr.substring(0, 4)}-${dateStr.substring(4, 6)}-${dateStr.substring(6, 8)}`;
    }
    return new Date().toISOString().split('T')[0];
  };

  // Generowanie opisu na podstawie nazwy modelu
  const generateDescription = (modelName: string, algorithm: string): string => {
    return `Model ${algorithm} z pliku ${modelName}, zoptymalizowany pod kątem detekcji.`;
  };

  // Pobieranie danych z serwera
  useEffect(() => {
    let isMounted = true;

    const fetchModels = async () => {
      if (!apiUrl) return;

      try {
        if (isMounted) {
          setLoading(true);
        }

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

        clearTimeout(timeoutId1); // Wyczyść timeout po zakończeniu żądania

        if (!algorithmsResponse.ok) {
          throw new Error(`Błąd pobierania algorytmów: ${algorithmsResponse.status}`);
        }

        const algorithms: string[] = await algorithmsResponse.json();
        console.log('Pobrane algorytmy:', algorithms);

        const sectionsData: SectionData[] = [];
        for (const algorithm of algorithms) {
          try {
            // Tworzymy AbortController dla drugiego żądania
            const controller2 = new AbortController();
            const timeoutId2 = setTimeout(() => controller2.abort(), 10000); // Timeout 10 sekund

            const modelsResponse = await fetch(`${apiUrl}/detect_model_versions/${algorithm}`, {
              method: 'GET',
              signal: controller2.signal,
            });

            clearTimeout(timeoutId2); // Wyczyść timeout po zakończeniu żądania

            if (!modelsResponse.ok) {
              throw new Error(`Błąd pobierania modeli dla ${algorithm}: ${modelsResponse.status}`);
            }

            const models: string[] = await modelsResponse.json();
            console.log(`Modele dla ${algorithm}:`, models);

            const modelItems: ModelItem[] = models.map((modelName, index) => ({
              id: `${algorithm}-${index}`,
              name: modelName,
              date: extractDateFromModelName(modelName),
              description: generateDescription(modelName, algorithm),
            }));

            sectionsData.push({
              algorithm,
              data: modelItems,
            });
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
          setSections(sectionsData);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : String(error);
        console.log('Błąd podczas pobierania modeli:', errorMessage);
        if (isMounted) {
          Alert.alert('Błąd', `Nie udało się pobrać listy modeli: ${errorMessage}`);
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
  }, [apiUrl]);

  const renderModelItem = ({ item }: { item: ModelItem }) => (
    <TouchableOpacity
      style={styles.modelCard}
      onPress={() => {
        setSelectedModel(item.name);
        navigation.navigate('Home');
      }}
    >
      <Text style={styles.modelName}>{item.name}</Text>
      <Text style={styles.modelDate}>Data utworzenia: {item.date}</Text>
      <Text style={styles.modelDescription}>{item.description}</Text>
    </TouchableOpacity>
  );

  const renderSectionHeader = ({ section }: { section: SectionData }) => (
    <Text style={styles.sectionHeader}>{section.algorithm}</Text>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Lista modeli</Text>
      {loading ? (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#00A1D6" />
          <Text style={styles.loadingText}>Ładowanie modeli...</Text>
        </View>
      ) : sections.length === 0 ? (
        <Text style={styles.noModelsText}>Brak dostępnych modeli.</Text>
      ) : (
        <SectionList
          sections={sections}
          renderItem={renderModelItem}
          renderSectionHeader={renderSectionHeader}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.listContainer}
        />
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
    padding: 10,
  },
  title: {
    fontSize: 28,
    color: '#FFF',
    fontWeight: 'bold',
    textAlign: 'center',
    marginVertical: 20,
  },
  listContainer: {
    paddingBottom: 20,
  },
  sectionHeader: {
    fontSize: 22,
    color: '#FFD700',
    fontWeight: 'bold',
    backgroundColor: '#1E1E1E',
    padding: 10,
    marginTop: 10,
  },
  modelCard: {
    backgroundColor: '#1E1E1E',
    borderRadius: 10,
    padding: 15,
    marginVertical: 8,
    marginHorizontal: 10,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  modelName: {
    fontSize: 18,
    color: '#FFD700',
    fontWeight: 'bold',
    marginBottom: 5,
  },
  modelDate: {
    fontSize: 14,
    color: '#888',
    marginBottom: 5,
  },
  modelDescription: {
    fontSize: 14,
    color: '#FFF',
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
  noModelsText: {
    color: '#FFF',
    fontSize: 16,
    textAlign: 'center',
    marginTop: 20,
  },
});

export default ModelsScreen;