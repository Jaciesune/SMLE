import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, FlatList, Image, Dimensions } from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useFocusEffect } from '@react-navigation/native';

type AnalysisResult = {
  id: string;
  photosBefore: string[];
  photosAfter: string[];
  pipesDetected: number[];
  totalPipes: number;
  confidence: number;
  algorithm: string;
  model: string;
  timestamp: string;
};

const HistoryScreen: React.FC = () => {
  const [history, setHistory] = useState<AnalysisResult[]>([]);

  // Ładowanie historii z AsyncStorage
  const loadHistory = async () => {
    try {
      const storedHistory = await AsyncStorage.getItem('analysisHistory');
      if (storedHistory) {
        const parsedHistory = JSON.parse(storedHistory);
        // Filtruj wpisy, które mają poprawne dane
        const validHistory = parsedHistory.filter((item: AnalysisResult) => {
          return (
            item &&
            Array.isArray(item.photosBefore) &&
            Array.isArray(item.photosAfter) &&
            Array.isArray(item.pipesDetected) &&
            item.photosBefore.length === item.photosAfter.length &&
            item.photosBefore.length === item.pipesDetected.length
          );
        });
        setHistory(validHistory);
        console.log('Załadowano historię:', validHistory);
      } else {
        setHistory([]);
        console.log('Brak historii w AsyncStorage');
      }
    } catch (error) {
      console.log('Błąd podczas ładowania historii analizy:', error);
      setHistory([]);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  // Odświeżanie historii przy każdym wejściu do zakładki
  useFocusEffect(
    React.useCallback(() => {
      loadHistory();
    }, [])
  );

  const renderPhotoPair = (photoBefore: string, photoAfter: string, index: number, pipes: number) => (
    <View key={index} style={styles.imageContainer}>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Przed</Text>
        <Image source={{ uri: photoBefore }} style={styles.image} />
      </View>
      <View style={styles.imageWrapper}>
        <Text style={styles.imageLabel}>Po</Text>
        <View style={styles.imageOverlay}>
          <Image source={{ uri: photoAfter }} style={styles.image} />
          {/* Symulacja zaznaczonych rur - czerwony prostokąt */}
          <View style={styles.overlayBox} />
        </View>
      </View>
      <Text style={styles.statsText}>Zdjęcie {index + 1}: {pipes} rur</Text>
    </View>
  );

  const renderHistoryItem = ({ item }: { item: AnalysisResult }) => (
    <View style={styles.historyItem}>
      <Text style={styles.timestamp}>{new Date(item.timestamp).toLocaleString()}</Text>
      {/* Zabezpieczenie przed undefined w photosBefore */}
      {Array.isArray(item.photosBefore) && item.photosBefore.length > 0 ? (
        item.photosBefore.map((photoBefore, index) =>
          renderPhotoPair(
            photoBefore,
            item.photosAfter[index],
            index,
            item.pipesDetected[index]
          )
        )
      ) : (
        <Text style={styles.errorText}>Brak zdjęć dla tej analizy.</Text>
      )}
      <View style={styles.statsContainer}>
        <Text style={styles.statsText}>Łącznie: {item.totalPipes} rur</Text>
        <Text style={styles.statsText}>Algorytm: {item.algorithm}</Text>
        <Text style={styles.statsText}>Model: {item.model}</Text>
        <Text style={styles.statsText}>Pewność: {item.confidence}%</Text>
      </View>
    </View>
  );

  return (
    <View style={styles.container}>
      {history.length === 0 ? (
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
  imageOverlay: {
    position: 'relative',
  },
  image: {
    width: imageSize,
    height: imageSize,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#00A1D6',
  },
  overlayBox: {
    position: 'absolute',
    top: 20,
    left: 20,
    width: 50,
    height: 50,
    borderWidth: 2,
    borderColor: '#FF0000',
    backgroundColor: 'rgba(255, 0, 0, 0.2)',
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
});

export default HistoryScreen;