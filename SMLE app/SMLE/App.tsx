import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/MaterialIcons';
import HomeScreen from './screens/HomeScreen';
import ModelsScreen from './screens/ModelsScreen';
import SettingsScreen from './screens/SettingsScreen';
import HistoryScreen from './screens/HistoryScreen';
import { BottomTabParamList } from './types/navigation';
import ErrorBoundary from './ErrorBoundary';
import AsyncStorage from '@react-native-async-storage/async-storage';

const Tab = createBottomTabNavigator<BottomTabParamList>();

const App = () => {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [defaultModel, setDefaultModel] = useState<string | null>(null);

  useEffect(() => {
    const clearStorageIfUpdated = async () => {
      try {
        // Resetujemy flagę, aby wymusić ponowne czyszczenie
        await AsyncStorage.removeItem('hasClearedStorage_v2');
        const hasCleared = await AsyncStorage.getItem('hasClearedStorage_v2');
        if (!hasCleared) {
          console.log('Czyszczenie AsyncStorage po aktualizacji...');
          await AsyncStorage.clear();
          await AsyncStorage.setItem('hasClearedStorage_v2', 'true');
          console.log('AsyncStorage wyczyszczone.');
        }
      } catch (error) {
        console.log('Błąd podczas czyszczenia AsyncStorage:', error);
      }
    };

    clearStorageIfUpdated();
  }, []);

  return (
    <ErrorBoundary>
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            tabBarIcon: ({ color, size }) => {
              let iconName: string = 'home';
              if (route.name === 'Home') {
                iconName = 'home';
              } else if (route.name === 'Models') {
                iconName = 'list';
              } else if (route.name === 'Settings') {
                iconName = 'settings';
              } else if (route.name === 'Historia') {
                iconName = 'history';
              }
              return <Icon name={iconName} size={size} color={color} />;
            },
            tabBarActiveTintColor: '#00A1D6',
            tabBarInactiveTintColor: '#888',
            tabBarStyle: {
              backgroundColor: '#1E1E1E',
              borderTopColor: '#333',
            },
            headerStyle: {
              backgroundColor: '#1E1E1E',
            },
            headerTintColor: '#FFF',
            headerTitleStyle: {
              fontWeight: 'bold',
            },
          })}
        >
          <Tab.Screen name="Home" component={HomeScreen} />
          <Tab.Screen
            name="Models"
            children={() => <ModelsScreen setSelectedModel={setSelectedModel} />}
          />
          <Tab.Screen
            name="Settings"
            children={() => <SettingsScreen setDefaultModel={setDefaultModel} />}
          />
          <Tab.Screen name="Historia" component={HistoryScreen} />
        </Tab.Navigator>
      </NavigationContainer>
    </ErrorBoundary>
  );
};

export default App;