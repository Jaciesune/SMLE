import React, { useEffect, useState } from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Icon from 'react-native-vector-icons/MaterialIcons';
import HomeScreen from './screens/HomeScreen';
import ModelsScreen from './screens/ModelsScreen';
import SettingsScreen from './screens/SettingsScreen';
import HistoryScreen from './screens/HistoryScreen';
import { BottomTabParamList } from './types/navigation';

// Utwórz navigator z typami
const Tab = createBottomTabNavigator<BottomTabParamList>();

const App = () => {
  // Stan dla wybranego modelu (dla ModelsScreen)
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  // Stan dla domyślnego modelu (dla SettingsScreen)
  const [defaultModel, setDefaultModel] = useState<string | null>(null);


  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ color, size }) => {
            let iconName: string = 'home'; // Domyślna wartość
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
  );
};

export default App;