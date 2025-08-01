// Copyright(C) 1999-2025 National Technology & Engineering Solutions
// of Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
// NTESS, the U.S. Government retains certain rights in this software.
//
// See packages/seacas/LICENSE for details

#include "Ioss_DatabaseIO.h"
#include "Ioss_GroupingEntity.h"
#include "Ioss_Property.h"
#include "Ioss_Region.h"
#include "Ioss_Utils.h"
#include "Ioss_VariableType.h"
#include <cassert>
#include <cstddef>
#include <fmt/ostream.h>
#include <iostream>
#include <string>

#include "Ioss_CodeTypes.h"
#include "Ioss_EntityType.h"
#include "Ioss_Field.h"
#include "Ioss_FieldManager.h"
#include "Ioss_ParallelUtils.h"
#include "Ioss_PropertyManager.h"
#include "Ioss_State.h"

/** \brief Base class constructor adds "name" and "entity_count" properties to the entity.
 *
 *  \param[in] io_database The database associated with the entity.
 *  \param[in] my_name The entity name.
 *  \param[in] entity_cnt The number of subentities in the entity.
 *
 */
Ioss::GroupingEntity::GroupingEntity(Ioss::DatabaseIO *io_database, const std::string &my_name,
                                     int64_t entity_cnt)
    : entityName(my_name), database_(io_database), entityCount(entity_cnt),
      hash_(Ioss::Utils::hash(my_name))
{
  properties.add(Ioss::Property(this, "name", Ioss::Property::STRING));
  properties.add(Ioss::Property(this, "entity_count", Ioss::Property::INTEGER));
  properties.add(Ioss::Property(this, "attribute_count", Ioss::Property::INTEGER));

  if (my_name != "null_entity") {
    Ioss::Field::BasicType int_type = Ioss::Field::INTEGER;
    if (io_database != nullptr) {
      int_type = field_int_type();
    }
    fields.add(Ioss::Field("ids", int_type, "scalar", Ioss::Field::MESH, entity_cnt));
  }
}

Ioss::GroupingEntity::GroupingEntity(const Ioss::GroupingEntity &other)
    : properties(other.properties), fields(other.fields), entityName(other.entityName),
      entityCount(other.entityCount), attributeCount(other.attributeCount),
      entityState(other.entityState), hash_(other.hash_)
{
}

Ioss::GroupingEntity::~GroupingEntity()
{
  // Only deleted by owning entity (Ioss::Region)
  database_ = nullptr;
}

// Default implementation is to do nothing. Redefined in Ioss::Region
// to actually delete the database.
void Ioss::GroupingEntity::delete_database() {}

void Ioss::GroupingEntity::really_delete_database()
{
  delete database_;
  database_ = nullptr;
}

const Ioss::GroupingEntity *Ioss::GroupingEntity::contained_in() const
{
  if (database_ == nullptr) {
    return nullptr;
  }
  return database_->get_region();
}

std::string Ioss::GroupingEntity::generic_name() const
{
  int64_t id = get_optional_property("id", 0);
  return Ioss::Utils::encode_entity_name(short_type_string(), id);
}

bool Ioss::GroupingEntity::is_alias(const std::string &my_name) const
{
  Region *region = database_->get_region();
  return region->get_alias(my_name, type()) == entityName;
}

Ioss::DatabaseIO *Ioss::GroupingEntity::get_database() const
{
  assert(database_ != nullptr);
  return database_;
}

void Ioss::GroupingEntity::set_database(Ioss::DatabaseIO *io_database)
{
  assert(database_ == nullptr);   // Must be unset if we are setting it.
  assert(io_database != nullptr); // Must be set to valid value
  database_ = io_database;
}

void Ioss::GroupingEntity::reset_database(Ioss::DatabaseIO *io_database)
{
  assert(io_database != nullptr); // Must be set to valid value
  database_ = io_database;
}

// Discuss Data Object functions:
// ---Affect the containing object:
//    open(in string object_name, out ?)
//    close()
//    destroy()
// ---Affect what the object contains:
//    set(in string propertyname, in any property_value)
//    get(in string propertyname, out any property_value)
//    add(in string propertyname);
//    delete(in string propertyname)
//    describe(out vector<Ioss::Properties>)
//

/** \brief Get the current Ioss::State of the entity.
 *
 *  \returns The current state.
 */
Ioss::State Ioss::GroupingEntity::get_state() const { return entityState; }

/** \brief Calculate and get an implicit property.
 *
 *  These are calculated from data stored in the EntityBlock instead of having
 *  an explicit value assigned. An example would be 'element_block_count' for a region.
 *  Note that even though this is a pure virtual function, an implementation
 *  is provided to return properties that are common to all 'block'-type grouping entities.
 *  Derived classes should call 'GroupingEntity::get_implicit_property'
 *  if the requested property is not specific to their type.
 */
Ioss::Property Ioss::GroupingEntity::get_implicit_property(const std::string &my_name) const
{
  // Handle properties generic to all GroupingEntities.
  // These include:
  if (my_name == "entity_count") {
    return {my_name, entityCount};
  }
  if (my_name == "name") {
    return {my_name, entityName};
  }
  if (my_name == "attribute_count") {
    count_attributes();
    return {my_name, static_cast<int>(attributeCount)};
  }

  // End of the line. No property of this name exists.
  IOSS_ERROR(fmt::format("\nERROR: Property '{}' does not exist on {} {}\n\n", my_name,
                         type_string(), name()));
}

bool Ioss::GroupingEntity::check_for_duplicate(const Ioss::Field &new_field) const
{
  // See if a field with the same name exists...
  if (field_exists(new_field.get_name())) {
    auto behavior = get_database()->get_duplicate_field_behavior();
    if (behavior != DuplicateFieldBehavior::IGNORE_) {
      // Get the existing field so we can compare with `new_field`
      const Ioss::Field &field = fields.getref(new_field.get_name());
      if (field != new_field) {
        std::string        warn_err = behavior == DuplicateFieldBehavior::WARNING_ ? "" : "ERROR: ";
        std::ostringstream errmsg;
        fmt::print(errmsg,
                   "{}Duplicate incompatible fields named '{}' on {} {}:\n"
                   "\tExisting  field: {} {} of size {} bytes with role '{}' and storage '{}',\n"
                   "\tDuplicate field: {} {} of size {} bytes with role '{}' and storage '{}'.",
                   warn_err, new_field.get_name(), type_string(), name(), field.raw_count(),
                   field.type_string(), field.get_size(), field.role_string(),
                   field.raw_storage()->name(), new_field.raw_count(), new_field.type_string(),
                   new_field.get_size(), new_field.role_string(), new_field.raw_storage()->name());
        if (behavior == DuplicateFieldBehavior::WARNING_) {
          auto util = get_database()->util();
          if (util.parallel_rank() == 0) {
            fmt::print(Ioss::WarnOut(), "{}\n", errmsg.str());
          }
          return true;
        }
        IOSS_ERROR(errmsg);
      }
    }
  }
  return false;
}

/** \brief Add a field to the entity's field manager.
 *
 *  Assumes that a field with the same name does not already exist.
 *
 *  \param[in] new_field The field to add
 *
 */
void Ioss::GroupingEntity::field_add(Ioss::Field new_field)
{
  size_t field_size = new_field.raw_count();

  if (new_field.get_role() == Ioss::Field::REDUCTION) {
    if (field_size == 0) {
      new_field.reset_count(1);
    }
    if (!check_for_duplicate(new_field)) {
      fields.add(new_field);
    }
    return;
  }

  size_t entity_size = entity_count();
  if (field_size == 0 && entity_size != 0) {
    // Set field size to match entity size...
    new_field.reset_count(entity_size);
  }
  else if (entity_size != field_size && type() != REGION) {
    std::string filename = get_database()->get_filename();
    IOSS_ERROR(fmt::format(
        "IO System error: The {} '{}' has a size of {},\nbut the field '{}' which is being "
        "output on that entity has a size of {}\non database '{}'.\nThe sizes must match.  "
        "This is an application error that should be reported.",
        type_string(), name(), entity_size, new_field.get_name(), field_size, filename));
  }
  if (!check_for_duplicate(new_field)) {
    fields.add(new_field);
  }
}

/** \brief Read field data from the database file into memory using a pointer.
 *
 *  \param[in] field_name The name of the field to read.
 *  \param[out] data The data.
 *  \param[in] data_size The number of bytes of data to be read.
 *  \returns The number of values read.
 *
 */
int64_t Ioss::GroupingEntity::get_field_data(const std::string &field_name, void *data,
                                             size_t data_size) const
{
  verify_field_exists(field_name, "input");

  Ioss::Field field  = get_field(field_name);
  int64_t     retval = internal_get_field_data(field, data, data_size);

  // At this point, transform the field if specified...
  if (retval >= 0) {
    field.transform(data);
  }

  return retval;
}

/** Zero-copy API.  *IF* a field is zero-copyable, then this function will set the `data`
 * pointer to point to a chunk of memory of size `data_size` bytes containing the field
 * data for the specified field.  If the field is not zero-copyable, then the  `data`
 * pointer will point to `nullptr` and `data_size` will be 0 and `retval` will be -2.
 * TODO: Verify that returning `-2` on error makes sense or helps at all...
 */
int64_t Ioss::GroupingEntity::get_field_data(const std::string &field_name, void **data,
                                             size_t *data_size) const
{
  verify_field_exists(field_name, "input");

  int64_t     retval = -1;
  Ioss::Field field  = get_field(field_name);
  if (field.zero_copy_enabled()) {
    retval = internal_get_zc_field_data(field, data, data_size);
  }
  else {
    retval     = -2;
    *data      = nullptr;
    *data_size = 0;
  }
  return retval;
}

/** \brief Write field data from memory into the database file using a pointer.
 *
 *  \param[in] field_name The name of the field to write.
 *  \param[in] data The data.
 *  \param[in] data_size The number of bytes of data to be written.
 *  \returns The number of values written.
 *
 */
int64_t Ioss::GroupingEntity::put_field_data(const std::string &field_name, void *data,
                                             size_t data_size) const
{
  verify_field_exists(field_name, "input");

  Ioss::Field field = get_field(field_name);
  field.transform(data);
  return internal_put_field_data(field, data, data_size);
}

/** \brief Get the number of fields with the given role (MESH, ATTRIBUTE, TRANSIENT, REDUCTION,
 * etc.)
 *         in the entity's field manager.
 *
 *  \returns The number of fields with the given role.
 */
size_t Ioss::GroupingEntity::field_count(Ioss::Field::RoleType role) const
{
  IOSS_FUNC_ENTER(m_);
  Ioss::NameList names = field_describe(role);
  return names.size();
}

void Ioss::GroupingEntity::count_attributes() const
{
  if (attributeCount > 0) {
    return;
  }

  // If the set has a field named "attribute", then the number of
  // attributes is equal to the component count of that field...
  Ioss::NameList results_fields = field_describe(Ioss::Field::ATTRIBUTE);

  Ioss::NameList::const_iterator IF;
  int64_t                        attribute_count = 0;
  for (IF = results_fields.begin(); IF != results_fields.end(); ++IF) {
    const std::string &field_name = *IF;
    if (field_name != "attribute" || results_fields.size() == 1) {
      Ioss::Field field = get_field(field_name);
      attribute_count += field.raw_storage()->component_count();
    }
  }
  attributeCount = attribute_count;
}

void Ioss::GroupingEntity::verify_field_exists(const std::string &field_name,
                                               const std::string &inout) const
{
  if (!field_exists(field_name)) {
    std::string filename = get_database()->get_filename();
    IOSS_ERROR(
        fmt::format("\nERROR: On database '{}', Field '{}' does not exist for {} on {} {}\n\n",
                    filename, field_name, inout, type_string(), name()));
  }
}

void Ioss::GroupingEntity::property_update(const std::string &property, int64_t value) const
{
  if (property_exists(property)) {
    if (get_property(property).get_int() != value) {
      auto *nge = const_cast<Ioss::GroupingEntity *>(this);
      nge->property_erase(property);
      nge->property_add(Ioss::Property(property, value));
    }
  }
  else {
    auto *nge = const_cast<Ioss::GroupingEntity *>(this);
    nge->property_add(Ioss::Property(property, value));
  }
}

void Ioss::GroupingEntity::property_update(const std::string &property,
                                           const std::string &value) const
{
  if (property_exists(property)) {
    if (get_property(property).get_string() != value) {
      auto *nge = const_cast<Ioss::GroupingEntity *>(this);
      nge->property_erase(property);
      nge->property_add(Ioss::Property(property, value));
    }
  }
  else {
    auto *nge = const_cast<Ioss::GroupingEntity *>(this);
    nge->property_add(Ioss::Property(property, value));
  }
}

bool Ioss::GroupingEntity::equal_(const Ioss::GroupingEntity &rhs, bool quiet) const
{
  bool same = true;
  if (this->entityName != rhs.entityName) {
    if (quiet) {
      return false;
    }
    fmt::print(Ioss::OUTPUT(), "GroupingEntity: entityName mismatch ({} vs. {})\n",
               this->entityName, rhs.entityName);
    same = false;
  }

  if (this->entityCount != rhs.entityCount) {
    if (quiet) {
      return false;
    }
    fmt::print(Ioss::OUTPUT(), "GroupingEntity: entityCount mismatch ([] vs. [])\n",
               this->entityCount, rhs.entityCount);
    same = false;
  }

  if (this->attributeCount != rhs.attributeCount) {
    if (quiet) {
      return false;
    }
    fmt::print(Ioss::OUTPUT(), "GroupingEntity: attributeCount mismatch ([] vs. [])\n",
               this->attributeCount, rhs.attributeCount);
    same = false;
  }

  if (this->entityState != rhs.entityState) {
    if (quiet) {
      return false;
    }
    fmt::print(Ioss::OUTPUT(), "GroupingEntity: entityState mismatch ([] vs. [])\n",
               static_cast<int>(this->entityState), static_cast<int>(rhs.entityState));
    same = false;
  }

  if (this->hash_ != rhs.hash_) {
    if (quiet) {
      return false;
    }
    fmt::print(Ioss::OUTPUT(), "GroupingEntity: hash_ mismatch ({} vs. {})\n", this->hash_,
               rhs.hash_);
    same = false;
  }

  /* COMPARE properties */
  Ioss::NameList lhs_properties = this->property_describe();
  Ioss::NameList rhs_properties = rhs.property_describe();

  if (lhs_properties.size() != rhs_properties.size()) {
    if (quiet) {
      return false;
    }
    fmt::print(Ioss::OUTPUT(), "GroupingEntity: NUMBER of properties are different ({} vs. {})\n",
               lhs_properties.size(), rhs_properties.size());
    same = false;
  }

  for (auto &lhs_property : lhs_properties) {
    auto it = std::find(rhs_properties.begin(), rhs_properties.end(), lhs_property);
    if (it == rhs_properties.end()) {
      if (!quiet) {
        fmt::print(Ioss::OUTPUT(), "WARNING: {}: INPUT property ({}) not found in input #2\n",
                   name(), lhs_property);
        same = false;
      }
      continue;
    }

    if (lhs_property == "IOSS_INTERNAL_CONTAINED_IN") {
      continue;
    }

    if (this->properties.get(lhs_property) != rhs.properties.get(lhs_property)) {
      // EMPIRICALLY, different representations (e.g., CGNS vs. Exodus) of the same mesh
      // can have different values for the "original_block_order" property.
      if (lhs_property == "original_block_order") {
        if (!quiet) {
          fmt::print(Ioss::OUTPUT(),
                     "WARNING: {}: values for \"original_block_order\" DIFFER ({} vs. {})\n",
                     name(), this->properties.get(lhs_property).get_int(),
                     rhs.properties.get(lhs_property).get_int());
        }
      }
      else {
        if (!quiet) {
          auto lhs_prop = this->properties.get(lhs_property);
          auto rhs_prop = rhs.properties.get(lhs_property);
          if (lhs_prop.get_type() == Ioss::Property::STRING) {
            auto p1_value = lhs_prop.get_string();
            auto p2_value = rhs_prop.get_string();
            fmt::print(Ioss::OUTPUT(),
                       "{}: PROPERTY value mismatch [STRING] ({}): ('{}' vs '{}')\n", name(),
                       lhs_property, p1_value, p2_value);
          }
          else if (lhs_prop.get_type() == Ioss::Property::INTEGER) {
            fmt::print(Ioss::OUTPUT(), "{}: PROPERTY value mismatch [INTEGER] ({}): ({} vs {})\n",
                       name(), lhs_property, lhs_prop.get_int(), rhs_prop.get_int());
          }
          else {
            fmt::print(Ioss::OUTPUT(), "{}: PROPERTY value mismatch ({}): unsupported type\n",
                       name(), lhs_property);
          }
        }
        else {
          return false;
        }
      }
      same = false;
    }
  }

  if (!quiet) {
    for (auto &rhs_property : rhs_properties) {
      auto it = std::find(lhs_properties.begin(), lhs_properties.end(), rhs_property);
      if (it == lhs_properties.end()) {
        fmt::print(Ioss::OUTPUT(), "WARNING: {}: OUTPUT property ({}) not found in input #1\n",
                   name(), rhs_property);
        same = false;
      }
    }
  }

  /* COMPARE fields */
  Ioss::NameList lhs_fields = this->field_describe();
  Ioss::NameList rhs_fields = rhs.field_describe();

  if (lhs_fields.size() != rhs_fields.size()) {
    if (!quiet) {
      fmt::print(Ioss::OUTPUT(), "\n{}: NUMBER of fields is different ({} vs. {})\n", name(),
                 lhs_fields.size(), rhs_fields.size());
      same = false;
    }
    else {
      return false;
    }
  }

  for (auto &field : lhs_fields) {
    if (!quiet) {
      const auto &f1 = this->fields.get(field);
      if (rhs.field_exists(field)) {
        const auto &f2 = rhs.fields.get(field);
        if (!f1.equal(f2)) {
          fmt::print(Ioss::OUTPUT(), "{}: FIELD ({}) mismatch\n\n", name(), field);
          same = false;
        }
      }
      else {
        fmt::print(Ioss::OUTPUT(), "{}: FIELD ({}) not found in input #2\n", name(), field);
        same = false;
      }
    }
    else {
      if (!this->fields.get(field).equal(rhs.fields.get(field))) {
        return false;
      }
    }
  }

  // See which fields are missing from input #1...
  // NOTE: `quiet` mode has already exited by this point.
  for (auto &field : rhs_fields) {
    if (!this->field_exists(field)) {
      fmt::print(Ioss::OUTPUT(), "{}: FIELD ({}) not found in input #1\n", name(), field);
      same = false;
    }
  }
  return same;
}

bool Ioss::GroupingEntity::operator==(const Ioss::GroupingEntity &rhs) const
{
  return equal_(rhs, true);
}

bool Ioss::GroupingEntity::operator!=(const Ioss::GroupingEntity &rhs) const
{
  return !(*this == rhs);
}

bool Ioss::GroupingEntity::equal(const Ioss::GroupingEntity &rhs) const
{
  return equal_(rhs, false);
}
