/*
  Copyright 2019 Equinor AS.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <config.h>
#include <opm/common/utility/MemPacker.hpp>

#include <opm/input/eclipse/EclipseState/IOConfig/FIPConfig.hpp>

namespace Opm {
namespace Serialization {
namespace detail {

//! \brief Specialization for std::bitset
template <std::size_t Size>
std::size_t Packing<false,std::bitset<Size>>::
packSize(const std::bitset<Size>& data)
{
    return Packing<true,unsigned long long>::packSize(data.to_ullong());
}

template <std::size_t Size>
void Packing<false,std::bitset<Size>>::
pack(const std::bitset<Size>& data,
     std::vector<char>& buffer, std::size_t& position)
{
    Packing<true,unsigned long long>::pack(data.to_ullong(), buffer, position);
}

template <std::size_t Size>
void Packing<false,std::bitset<Size>>::
unpack(std::bitset<Size>& data,
       const std::vector<char>& buffer, std::size_t& position)
{
    unsigned long long d;
    Packing<true,unsigned long long>::unpack(d, buffer, position);
    data = std::bitset<Size>(d);
}

std::size_t Packing<false,std::string>::
packSize(const std::string& data)
{
    return sizeof(std::size_t) + data.size();
}

void Packing<false,std::string>::
pack(const std::string& data,
     std::vector<char>& buffer, std::size_t& position)
{
    Packing<true,std::size_t>::pack(data.size(), buffer, position);
    Packing<true,char>::pack(data.data(), data.size(), buffer, position);
}

void Packing<false,std::string>::
unpack(std::string& data, const std::vector<char>& buffer, std::size_t& position)
{
    std::size_t length = 0;
    Packing<true,std::size_t>::unpack(length, buffer, position);
    std::vector<char> cStr(length+1, '\0');
    Packing<true,char>::unpack(cStr.data(), length, buffer, position);
    data.clear();
    data.append(cStr.data(), length);
}

std::size_t Packing<false,time_point>::
packSize(const time_point&)
{
    return Packing<true, time_point::duration::rep>::packSize(time_point::duration::rep());
}

void Packing<false,time_point>::
pack(const time_point& data,
     std::vector<char>& buffer, std::size_t& position)
{
    Packing<true, time_point::duration::rep>::pack(data.time_since_epoch().count(), buffer, position);
}

void Packing<false,time_point>::
unpack(time_point& data, const std::vector<char>& buffer, std::size_t& position)
{
    time_point::duration::rep res;
    Packing<true, time_point::duration::rep>::unpack(res, buffer, position);
    data = time_point(time_point::duration(res));
}

template struct Packing<false,std::bitset<3>>;
template struct Packing<false,std::bitset<4>>;
template struct Packing<false,std::bitset<10>>;
constexpr int NumFip = static_cast<int>(FIPConfig::OutputField::NUM_FIP_REPORT);
template struct Packing<false,std::bitset<NumFip>>;

}

} // end namespace Serialization
} // end namespace Opm
