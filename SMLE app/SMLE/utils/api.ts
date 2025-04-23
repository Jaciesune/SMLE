import AsyncStorage from '@react-native-async-storage/async-storage';
import DeviceInfo from 'react-native-device-info';
import { Platform } from 'react-native';

export const getApiUrl = async (): Promise<string> => {
  try {
    const storedIp = await AsyncStorage.getItem('serverIp');
    if (storedIp) {
      return storedIp;
    }

    // Domyślne IP dla emulatora lub fizycznego urządzenia
    const defaultIp = Platform.OS === 'android' && DeviceInfo.isEmulatorSync()
      ? 'http://10.0.2.2:8000'
      : 'http://192.168.0.3:8000';

    return defaultIp;
  } catch (error) {
    console.log('Błąd podczas wczytywania API_URL:', error);
    return 'http://192.168.0.3:8000'; // Domyślne IP w razie błędu
  }
};