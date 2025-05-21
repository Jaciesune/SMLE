import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, FlatList, Image, Dimensions, Alert, ActivityIndicator } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';
import ReactNativeBlobUtil from 'react-native-blob-util';

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

type PhotoPairProps = {
  photoBefore: string;
  photoAfter: string;
  index: number;
  pipes: number;
};

const PhotoPair: React.FC<PhotoPairProps> = ({ photoBefore, photoAfter, index, pipes }) => {
  const [beforeUriValid, setBeforeUriValid] = useState<boolean>(true); // Zakładamy, że URI jest poprawne na start
  const [afterUriValid, setAfterUriValid] = useState<boolean>(true);

  return (
    <View key={index} style={styles.imageContainer}>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Przed</Text>
        {beforeUriValid ? (
          <Image
            source={{ uri: photoBefore }}
            style={styles.image}
            onError={(e) => {
              console.log(`Błąd ładowania zdjęcia Before ${index + 1}:`, e.nativeEvent.error);
              setBeforeUriValid(false);
            }}
          />
        ) : (
          <Text style={styles.errorText}>Zdjęcie niedostępne</Text>
        )}
      </View>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Po</Text>
        {afterUriValid ? (
          <Image
            source={{ uri: photoAfter }}
            style={styles.image}
            onError={(e) => {
              console.log(`Błąd ładowania zdjęcia After ${index + 1}:`, e.nativeEvent.error);
              setAfterUriValid(false);
            }}
          />
        ) : (
          <Text style={styles.errorText}>Zdjęcie niedostępne</Text>
        )}
      </View>
      <Text style={styles.statsText}>Zdjęcie {index + 1}: {pipes} elementów</Text>
    </View>
  );
};

const HistoryScreen: React.FC = () => {
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const storedHistory = await AsyncStorage.getItem('analysisHistory');
      console.log('Surowa historia z AsyncStorage:', storedHistory);

      if (storedHistory) {
        let parsedHistory: AnalysisResult[];
        try {
          parsedHistory = JSON.parse(storedHistory);
        } catch (parseError) {
          console.log('Błąd parsowania JSON:', parseError);
          throw new Error('Nieprawidłowy format danych w AsyncStorage.');
        }
        console.log('Sparowana historia:', parsedHistory);

        const validHistory = parsedHistory.filter((item: AnalysisResult) => {
          const isValid =
            item &&
            Array.isArray(item.photosBefore) &&
            Array.isArray(item.photosAfter) &&
            Array.isArray(item.pipesDetected) &&
            item.photosBefore.length === item.photosAfter.length &&
            item.photosBefore.length === item.pipesDetected.length &&
            typeof item.totalPipes === 'number' &&
            typeof item.algorithm === 'string' &&
            typeof item.model === 'string' &&
            typeof item.timestamp === 'string';
          if (!isValid) {
            console.log('Nieprawidłowy wpis w historii:', item);
          }
          return isValid;
        });
        setHistory(validHistory);
        console.log('Załadowano historię:', validHistory);
      } else {
        setHistory([]);
        console.log('Brak historii w AsyncStorage');
      }
    } catch (error) {
      console.log('Błąd podczas ładowania historii analizy:', error);
      Alert.alert('Błąd', 'Nie udało się wczytać historii. Spróbuj ponownie.');
      setHistory([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  useFocusEffect(
    useCallback(() => {
      loadHistory();
    }, [])
  );

  const renderHistoryItem = ({ item }: { item: AnalysisResult }) => {
    console.log('Renderowanie elementu historii:', item);
    return (
      <View style={styles.historyItem}>
        <Text style={styles.timestamp}>{new Date(item.timestamp).toLocaleString()}</Text>
        {Array.isArray(item.photosBefore) && item.photosBefore.length > 0 ? (
          item.photosBefore.map((photoBefore, index) => (
            <PhotoPair
              key={index}
              photoBefore={photoBefore}
              photoAfter={item.photosAfter[index]}
              index={index}
              pipes={item.pipesDetected[index]}
            />
          ))
        ) : (
          <Text style={styles.errorText}>Brak zdjęć dla tej analizy.</Text>
        )}
        <View style={styles.statsContainer}>
          <Text style={styles.statsText}>Łącznie: {item.totalPipes} elementów</Text>
          <Text style={styles.statsText}>Algorytm: {item.algorithm}</Text>
          <Text style={styles.statsText}>Model: {item.model}</Text>
        </View>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {loading ? (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color="#00A1D6" />
          <Text style={styles.loadingText}>Ładowanie historii...</Text>
        </View>
      ) : history.length === 0 ? (
        <Text style={styles.emptyText}>Brak wyników w historii.</Text>
      ) : (
        <FlatList
          data={history}
          renderItem={renderHistoryItem}
          keyExtractor={(item) => item.id}
          contentContainerStyle={styles.list}
        />
      )}
    </View>
  );
};

const { width } = Dimensions.get('window');
const imageSize = (width - 60) / 2;

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#121212',
    padding: 10,
  },
  historyItem: {
    backgroundColor: '#1E1E1E',
    borderRadius: 10,
    padding: 15,
    marginBottom: 15,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.3,
    shadowRadius: 4,
  },
  timestamp: {
    color: '#FFD700',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  imageContainer: {
    marginBottom: 10,
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
  image: {
    width: imageSize,
    height: imageSize,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#00A1D6',
  },
  statsContainer: {
    marginTop: 10,
  },
  statsText: {
    color: '#FFF',
    fontSize: 16,
    marginBottom: 5,
  },
  errorText: {
    color: '#FF4444',
    fontSize: 16,
    marginBottom: 10,
  },
  emptyText: {
    color: '#888',
    fontSize: 18,
    textAlign: 'center',
    marginTop: 20,
  },
  list: {
    paddingBottom: 20,
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

export default HistoryScreen;