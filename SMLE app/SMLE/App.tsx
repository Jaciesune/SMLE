import React, { useState, useEffect } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import HomeScreen from './screens/HomeScreen';
import ModelsScreen from './screens/ModelsScreen';
import SettingsScreen from './screens/SettingsScreen';
import Icon from 'react-native-vector-icons/Ionicons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const Tab = createBottomTabNavigator();

const App: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  // Ładowanie domyślnego modelu przy starcie aplikacji
  useEffect(() => {
    const loadDefaultModel = async () => {
      try {
        const storedModel = await AsyncStorage.getItem('defaultModel');
        setSelectedModel(storedModel || null);
      } catch (error) {
        console.log('Błąd podczas ładowania domyślnego modelu:', error);
      }
    };
    loadDefaultModel();
  }, []);

  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ focused, color, size }) => {
            let iconName: string;

            if (route.name === 'Home') {
              iconName = focused ? 'home' : 'home-outline';
            } else if (route.name === 'Models') {
              iconName = focused ? 'cube' : 'cube-outline';
            } else if (route.name === 'Settings') {
              iconName = focused ? 'settings' : 'settings-outline';
            } else {
              iconName = 'help-circle-outline';
            }

            return <Icon name={iconName} size={size} color={color} />;
          },
          tabBarStyle: {
            backgroundColor: '#1E1E1E',
            borderTopColor: '#333',
            paddingBottom: 5,
            paddingTop: 5,
          },
          tabBarActiveTintColor: '#FFD700',
          tabBarInactiveTintColor: '#888',
          headerStyle: {
            backgroundColor: '#1E1E1E',
          },
          headerTintColor: '#FFF',
        })}
      >
        <Tab.Screen name="Home">
          {(props) => <HomeScreen {...props} selectedModel={selectedModel} />}
        </Tab.Screen>
        <Tab.Screen name="Models">
          {(props) => <ModelsScreen {...props} setSelectedModel={setSelectedModel} />}
        </Tab.Screen>
        <Tab.Screen name="Settings">
          {(props) => <SettingsScreen {...props} setDefaultModel={setSelectedModel} />}
        </Tab.Screen>
      </Tab.Navigator>
    </NavigationContainer>
  );
};

export default App;