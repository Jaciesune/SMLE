import React from 'react';
import { View, Text, SectionList, StyleSheet, TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';

const mockModels = [
  {
    algorithm: 'MCNN',
    data: [
      {
        id: '1',
        name: 'DrugyTestMCNN_20250404_173029_checkpoint.pth',
        date: '2025-04-04',
        description: 'Model MCNN do analizy obrazów, zoptymalizowany pod kątem precyzji.',
      },
    ],
  },
  {
    algorithm: 'YOLO',
    data: [
      {
        id: '2',
        name: 'ModelYOLO_20250405_123456.pth',
        date: '2025-04-05',
        description: 'Model YOLO v5, szybki i skuteczny w detekcji obiektów.',
      },
      {
        id: '4',
        name: 'YOLOv8_20250407_654321.pth',
        date: '2025-04-07',
        description: 'Model YOLO v8, najnowsza wersja z poprawioną dokładnością.',
      },
    ],
  },
  {
    algorithm: 'Faster R-CNN',
    data: [
      {
        id: '3',
        name: 'FasterRCNN_20250406_789012.pth',
        date: '2025-04-06',
        description: 'Model Faster R-CNN, idealny do złożonych analiz obrazów.',
      },
    ],
  },
];

type Props = {
  setSelectedModel: (model: string) => void;
};

const ModelsScreen: React.FC<Props> = ({ setSelectedModel }) => {
  const navigation = useNavigation();

  const renderModelItem = ({ item }: { item: typeof mockModels[0]['data'][0] }) => (
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

  const renderSectionHeader = ({ section }: { section: typeof mockModels[0] }) => (
    <Text style={styles.sectionHeader}>{section.algorithm}</Text>
  );

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Lista modeli</Text>
      <SectionList
        sections={mockModels}
        renderItem={renderModelItem}
        renderSectionHeader={renderSectionHeader}
        keyExtractor={(item) => item.id}
        contentContainerStyle={styles.listContainer}
      />
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
});

export default ModelsScreen;